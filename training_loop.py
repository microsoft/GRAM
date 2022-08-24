
import os
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from generators import generators
from discriminators import discriminators
from processes import processes
import configs as configs
import datasets
from tqdm import tqdm

from torch_ema import ExponentialMovingAverage


def set_generator(config, device, opt):
    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )

    if opt.load_dir != '':
        generator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_generator.pth'%opt.set_step), map_location='cpu'))

    generator = generator.to(device)
    
    if opt.load_dir != '':
        ema = torch.load(os.path.join(opt.load_dir, 'step%06d_ema.pth'%opt.set_step), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'step%06d_ema2.pth'%opt.set_step), map_location=device)
    else:
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
    
    return generator, ema, ema2


def set_discriminator(config, device, opt):
    discriminator = getattr(discriminators, config['discriminator']['class'])(**config['discriminator']['kwargs'])
    if opt.load_dir != '':
        discriminator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_discriminator.pth'%opt.set_step), map_location='cpu'))
    
    discriminator = discriminator.to(device)
    
    return discriminator


def set_optimizer_G(generator_ddp, config, opt):
    param_groups = []
    if 'mapping_network_lr' in config['optimizer']:
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if 'module.representation.rf_network.mapping_network' in n]
        param_groups.append({'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':config['optimizer']['mapping_network_lr']})
    if 'sampling_network_lr' in config['optimizer']:
        sampling_network_parameters = [p for n, p in generator_ddp.named_parameters() if 'module.representation.sample_network' in n]
        param_groups.append({'params': sampling_network_parameters, 'name': 'sampling_network', 'lr':config['optimizer']['sampling_network_lr']})
    generator_parameters = [p for n, p in generator_ddp.named_parameters() if 
        ('mapping_network_lr' not in config['optimizer'] or 'module.representation.rf_network.mapping_network' not in n) and
        ('sampling_network_lr' not in config['optimizer'] or 'module.representation.sample_network' not in n)]
    param_groups.append({'params': generator_parameters, 'name': 'generator'})
    optimizer_G = torch.optim.Adam(param_groups, lr=config['optimizer']['gen_lr'], betas=config['optimizer']['betas'])

    if opt.load_dir != '':
        state_dict = torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_G.pth'%opt.set_step), map_location='cpu')
        optimizer_G.load_state_dict(state_dict)
    
    return optimizer_G


def set_optimizer_D(discriminator_ddp, config, opt):
    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=config['optimizer']['disc_lr'], betas=config['optimizer']['betas'])

    if opt.load_dir != '':
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_D.pth'%opt.set_step), map_location='cpu'))
    
    return optimizer_D


def training_process(rank, world_size, opt, device):
#--------------------------------------------------------------------------------------
# extract training config
    config = getattr(configs, opt.config)
    if opt.patch_split is not None:
        config['process']['kwargs']['patch_split'] = opt.patch_split

#--------------------------------------------------------------------------------------
# set amp gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    if opt.load_dir != '':
        if not config['global'].get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_scaler.pth'%opt.set_step)))

    if config['global'].get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)


#--------------------------------------------------------------------------------------
#set generator and discriminator

    generator, ema, ema2 = set_generator(config, device, opt)
    discriminator = set_discriminator(config, device, opt)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)

    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    if rank == 0:
        for name, param in generator_ddp.named_parameters():
            print(f'{name:<{96}}{param.shape}')
        total_num = sum(p.numel() for p in generator_ddp.parameters())
        trainable_num = sum(p.numel() for p in generator_ddp.parameters() if p.requires_grad)
        print('G: Total ', total_num, ' Trainable ', trainable_num)
        
        for name, param in discriminator_ddp.named_parameters():
            print(f'{name:<{96}}{param.shape}')
        total_num = sum(p.numel() for p in discriminator_ddp.parameters())
        trainable_num = sum(p.numel() for p in discriminator_ddp.parameters() if p.requires_grad)
        print('D: Total ', total_num, ' Trainable ', trainable_num)

#--------------------------------------------------------------------------------------
# set optimizers
    optimizer_G = set_optimizer_G(generator_ddp, config, opt)
    optimizer_D = set_optimizer_D(discriminator_ddp, config, opt)

    torch.cuda.empty_cache()

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    # ----------
    #  Training
    # ----------
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(opt.config))
        f.write('\n\n')
        f.write(str(config))

    torch.manual_seed(rank)
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True) # Keeps track of total training
    total_progress_bar.update(discriminator.epoch) # Keeps track of progress to next stage
    interior_step_bar = tqdm(desc = "Steps", dynamic_ncols=True)

#--------------------------------------------------------------------------------------
# set loss
    process = getattr(processes, config['process']['class'])(**config['process']['kwargs'])

#--------------------------------------------------------------------------------------
# get dataset
    dataset = getattr(datasets, config['dataset']['class'])(**config['dataset']['kwargs'])
    dataloader, CHANNELS = datasets.get_dataset_distributed_(
        dataset,
        world_size,
        rank,
        config['global']['batch_size']
    )

#--------------------------------------------------------------------------------------
# main training loop

    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)     
             
        #--------------------------------------------------------------------------------------
        # trainging iterations
        for i, (imgs, poses) in enumerate(dataloader):

            # save model
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                torch.save(ema, os.path.join(opt.output_dir, 'step%06d_ema.pth'%discriminator.step))
                torch.save(ema2, os.path.join(opt.output_dir, 'step%06d_ema2.pth'%discriminator.step))
                torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'step%06d_generator.pth'%discriminator.step))
                torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'step%06d_discriminator.pth'%discriminator.step))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'step%06d_optimizer_G.pth'%discriminator.step))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'step%06d_optimizer_D.pth'%discriminator.step))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'step%06d_scaler.pth'%discriminator.step))
                torch.cuda.empty_cache()
            dist.barrier()

            if scaler.get_scale() < 1:
                scaler.update(1.)

            real_imgs = imgs.to(device, non_blocking=True)
            real_poses = poses.to(device, non_blocking=True)

            generator_ddp.train()
            discriminator_ddp.train()
            
            #--------------------------------------------------------------------------------------
            # TRAIN DISCRIMINATOR
            d_loss = process.train_D(real_imgs, real_poses, generator_ddp, discriminator_ddp, optimizer_D, scaler, config, device)
            discriminator_losses.append(d_loss)

            # TRAIN GENERATOR
            g_loss = process.train_G(real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, scaler, config, device)
            generator_losses.append(g_loss)

            #--------------------------------------------------------------------------------------
            # print and save
            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Step: {discriminator.step}] [D loss: {d_loss}] [G loss: {g_loss}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [Img Size: {config['global']['img_size']}] [Batch Size: {config['global']['batch_size']}] [Scale: {scaler.get_scale()}]")

                # save fixed angle generated images
                if discriminator.step % opt.sample_interval == 0:
                    process.snapshot(generator_ddp, discriminator_ddp, config, opt.output_dir, device)

                # save_model
                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))
                torch.cuda.empty_cache()
            dist.barrier()
                            
            #--------------------------------------------------------------------------------------

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1