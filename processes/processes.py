import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


class Gan3DProcess:
    def __init__(self, batch_split=1, patch_split=None, real_pos_lambda=15., r1_lambda=1., pos_lambda=15., r1_interval=1) -> None:
        self.batch_split = batch_split
        self.patch_split = patch_split
        self.real_pos_lambda = real_pos_lambda
        self.r1_lambda = r1_lambda
        self.pos_lambda = pos_lambda
        self.r1_interval = r1_interval
        self.fixed_z = z_sampler((25, 256), device='cpu', dist='gaussian')

    def train_D(self, real_imgs, real_positions, generator_ddp, discriminator_ddp, optimizer_D, scaler, config, device):
        with torch.cuda.amp.autocast():
            # Generate images for discriminator training
            with torch.no_grad():
                z = z_sampler((real_imgs.shape[0], generator_ddp.module.z_dim), device=device, dist='gaussian')
                split_batch_size = z.shape[0] // self.batch_split
                gen_imgs = []
                gen_positions = []
                for split in range(self.batch_split):
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    if self.patch_split is None:
                        g_imgs, g_pos = generator_ddp(subset_z, **config['camera'])
                    else:
                        g_imgs = []
                        g_imgs_patch, g_pos = generator_ddp(subset_z, **config['camera'], patch=(0, self.patch_split))
                        g_imgs.append(g_imgs_patch)
                        for patch_idx in range(1, self.patch_split):
                            g_imgs_patch, _ = generator_ddp(subset_z, **config['camera'], camera_pos=g_pos, patch=(patch_idx, self.patch_split))
                            g_imgs.append(g_imgs_patch)
                        g_imgs = torch.cat(g_imgs,-1).reshape(split_batch_size,3,generator_ddp.module.img_size,generator_ddp.module.img_size)
                    gen_imgs.append(g_imgs)
                    gen_positions.append(g_pos)
                gen_imgs = torch.cat(gen_imgs, axis=0)
                gen_positions = torch.cat(gen_positions, axis=0)

            real_imgs.requires_grad = True
            r_preds, r_pred_position = discriminator_ddp(real_imgs)

        if self.r1_lambda > 0 and discriminator_ddp.module.step % self.r1_interval == 0:
            # Gradient penalty
            grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
            inv_scale = 1./scaler.get_scale()
            grad_real = [p * inv_scale for p in grad_real][0]

        with torch.cuda.amp.autocast():
            if self.r1_lambda > 0 and discriminator_ddp.module.step % self.r1_interval == 0:
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * self.r1_lambda * grad_penalty
            else:
                grad_penalty = 0
            
            g_preds, g_pred_position = discriminator_ddp(gen_imgs.detach())
            position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
            if self.real_pos_lambda > 0:
                position_penalty += nn.MSELoss()(r_pred_position, real_positions) * self.real_pos_lambda
            d_loss = F.softplus(g_preds).mean() + F.softplus(-r_preds).mean() + grad_penalty + position_penalty

        optimizer_D.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizer_D)
        nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
        scaler.step(optimizer_D)

        return d_loss.item()

    def train_G(self, real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, scaler, config, device):
        z = z_sampler((real_imgs.shape[0], generator_ddp.module.z_dim), device=device, dist='gaussian')
        split_batch_size = z.shape[0] // self.batch_split

        for split in range(self.batch_split):
            if self.patch_split is None:
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    gen_imgs, gen_positions = generator_ddp(subset_z, **config['camera'])
                    g_preds, g_pred_position = discriminator_ddp(gen_imgs)
                    position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
                    g_loss = F.softplus(-g_preds).mean() + position_penalty
                scaler.scale(g_loss).backward()
            else:
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    gen_imgs = []
                    with torch.no_grad():
                        gen_imgs_patch, gen_positions = generator_ddp(subset_z, **config['camera'], patch=(0, self.patch_split))
                        gen_imgs.append(gen_imgs_patch)
                        for patch_idx in range(1, self.patch_split):
                            gen_imgs_patch, _ = generator_ddp(subset_z, **config['camera'], camera_pos=gen_positions, patch=(patch_idx, self.patch_split))
                            gen_imgs.append(gen_imgs_patch)
                        gen_imgs = torch.cat(gen_imgs,-1).reshape(split_batch_size,3,generator_ddp.module.img_size,generator_ddp.module.img_size)
                    gen_imgs.requires_grad = True
                    g_preds, g_pred_position = discriminator_ddp(gen_imgs)
                    position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
                    g_loss = F.softplus(-g_preds).mean() + position_penalty

                grad_gen_imgs = torch.autograd.grad(outputs=scaler.scale(g_loss), inputs=gen_imgs, create_graph=False)[0]
                grad_gen_imgs = grad_gen_imgs.reshape(split_batch_size,3,-1)
                grad_gen_imgs = grad_gen_imgs.detach()

                for patch_idx in range(self.patch_split):
                    with torch.cuda.amp.autocast():
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        gen_imgs_patch, _ = generator_ddp(subset_z, **config['camera'], camera_pos=gen_positions, patch=(patch_idx, self.patch_split))
                    start = generator_ddp.module.img_size*generator_ddp.module.img_size*patch_idx//self.patch_split
                    end = generator_ddp.module.img_size*generator_ddp.module.img_size*(patch_idx+1)//self.patch_split
                    gen_imgs_patch.backward(grad_gen_imgs[...,start:end])

        scaler.unscale_(optimizer_G)
        nn.utils.clip_grad_norm_(generator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
        scaler.step(optimizer_G)
        scaler.update()
        optimizer_G.zero_grad()
        ema.update(generator_ddp.parameters())
        ema2.update(generator_ddp.parameters())

        return g_loss.item()

    def snapshot(self, generator_ddp, discriminator_ddp, config, output_dir, device, batchsize=1, patch_split=4):
        with torch.no_grad():
            generator_ddp.module.get_avg_w()

            gen_imgs = []
            for i in range(0, 25, batchsize):
                g_imgs = []
                for patch_idx in range(patch_split):
                    g_imgs_patch = generator_ddp.module.forward(self.fixed_z[i:(i+batchsize)].to(device), **config['camera'], camera_origin=[0., 0., 1.], truncation_psi=0.7, patch=(patch_idx, patch_split))[0]
                    g_imgs.append(g_imgs_patch)
                g_imgs = torch.cat(g_imgs,-1).reshape(batchsize,3,generator_ddp.module.img_size,generator_ddp.module.img_size)
                gen_imgs.append(g_imgs)
            gen_imgs = torch.cat(gen_imgs, dim=0)
            save_image(gen_imgs[:25], os.path.join(output_dir, "%06d_fixed.png"%discriminator_ddp.module.step), nrow=5, normalize=True, range=(-1, 1))

            gen_imgs = []
            for i in range(0, 25, batchsize):
                g_imgs = []
                for patch_idx in range(patch_split):
                    g_imgs_patch = generator_ddp.module.forward(self.fixed_z[i:(i+batchsize)].to(device), **config['camera'], camera_origin=[-np.sin(0.5), 0., np.cos(0.5)], truncation_psi=0.7, patch=(patch_idx, patch_split))[0]
                    g_imgs.append(g_imgs_patch)
                g_imgs = torch.cat(g_imgs,-1).reshape(batchsize,3,generator_ddp.module.img_size,generator_ddp.module.img_size)
                gen_imgs.append(g_imgs)
            gen_imgs = torch.cat(gen_imgs, dim=0)
            save_image(gen_imgs[:25], os.path.join(output_dir, "%06d_tilted.png"%discriminator_ddp.module.step), nrow=5, normalize=True, range=(-1, 1))
