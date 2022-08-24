import os
from pickle import FALSE
import shutil
import torch
import numpy as np

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from pytorch_fid import fid_score
import datasets
from tqdm import tqdm
import copy
import argparse
import imageio
import cv2

from generators import generators
import configs


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.png'), normalize=True, range=(-1, 1))
            img_counter += 1


def setup_evaluation(dataset_name, generated_dir, target_size=128, num_real_images=8000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, img_size=target_size)
        print('outputting real images...')
        output_real_images(dataloader, num_real_images, real_dir)
        print('...done')

    os.makedirs(generated_dir, exist_ok=True)
    return real_dir


def calculate_fid(dataset_name, generated_dir, target_size=128):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    for i in range(10):
        try:
            fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
            break
        except:
            print('failed to load evaluation images, try %02d times'%i)

    torch.cuda.empty_cache()

    return fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='./pretrained_models/FFHQ_default/generator.pth')
    parser.add_argument('--output_dir', type=str, default='./experiments/fid_evaluation/FFHQ_default')
    parser.add_argument('--config', type=str, default='FFHQ_default')
    parser.add_argument('--num_real_images', type=int, default=8000)
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--no_watermark', default=True, help='True to eliminate watermarks. default=True.', action='store_true')
    opt = parser.parse_args()

    config = getattr(configs, opt.config)

    os.makedirs(opt.output_dir, exist_ok=True)
    
    watermark = imageio.imread('images/watermark.png')
    watermark = cv2.resize(watermark,(config['global']['img_size'],config['global']['img_size']), interpolation = cv2.INTER_AREA).astype(np.float32)/255
    watermark = torch.tensor(watermark).unsqueeze(0).permute(0,3,1,2).to('cuda')

    real_images_dir = setup_evaluation(config['dataset']['class'], opt.output_dir, target_size=config['global']['img_size'], num_real_images=opt.num_real_images)

    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )

    generator.load_state_dict(torch.load(os.path.join(opt.generator_file), map_location='cpu'))
    generator = generator.to('cuda')
    generator.eval()
    
    ema = torch.load(os.path.join(opt.generator_file.replace('generator', 'ema')), map_location='cuda')
    parameters = [p for p in generator.parameters() if p.requires_grad]
    ema.copy_to(parameters)

    with torch.no_grad():
        generator.get_avg_w()
        for img_counter in tqdm(range(opt.num_images)):
            z = torch.randn(1, 256, device='cuda')
            with torch.no_grad():
                img = generator(z, **config['camera'])[0]
                if not opt.no_watermark:
                    img = img*(1-watermark[:,-1:]) + watermark[:,-1:]*(2*watermark[:,:-1] - 1)
                save_image(img, os.path.join(opt.output_dir, f'{img_counter:0>5}.png'), normalize=True, range=(-1, 1))

    metrics_dict = calculate_metrics(opt.output_dir, real_images_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    print(opt.generator_file)
    print(metrics_dict)
    with open(os.path.join(opt.output_dir, 'metrics.txt'), 'w') as f:
        f.write(str(metrics_dict))
