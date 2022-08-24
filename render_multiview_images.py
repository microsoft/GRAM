import argparse
import numpy as np
import torch
import math
import os
from tqdm import tqdm
from generators import generators
import configs
import imageio
import cv2
import re

def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='./pretrained_models/FFHQ_default/generator.pth')
    parser.add_argument('--output_dir', type=str, default='./experiments/multiview_images/FFHQ_default')
    parser.add_argument('--config', type=str, default='FFHQ_default')
    parser.add_argument('--seeds', type=parse_idx_range, default='0-19')
    parser.add_argument('--frames', type=str, default=30)
    parser.add_argument('--psi', type=str, default=0.7)
    parser.add_argument('--no_watermark', default=False, help='False to eliminate watermarks. default=True.', action='store_true')
    opt = parser.parse_args()

    config = getattr(configs, opt.config)

    os.makedirs(opt.output_dir, exist_ok=True)
   
    watermark = imageio.imread('images/watermark.png')
    watermark = cv2.resize(watermark,(config['global']['img_size'],config['global']['img_size']), interpolation = cv2.INTER_AREA).astype(np.float32)/255

    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )

    generator.load_state_dict(torch.load(os.path.join(opt.generator_file), map_location='cpu'),strict=False)
    generator = generator.to('cuda')
    generator.eval()
    
    ema = torch.load(os.path.join(opt.generator_file.replace('generator', 'ema')), map_location='cuda')
    parameters = [p for p in generator.parameters() if p.requires_grad]
    ema.copy_to(parameters)

    setting = 1 if opt.config != 'CARLA_default' else 2
    if setting == 1:     # FFHQ & CATS
        h_mean = math.pi * (90 / 180)
        v_mean = math.pi * (90 / 180)
        generator.renderer.lock_view_dependence = True
        frames = opt.frames
        concat = None
        face_yaws = list(np.linspace(-0.4, 0.4, frames // 2 + 1)[:-1]) + list(np.linspace(0.4, -0.4, frames // 2 + 1)[:-1])
        face_pitchs = [0*np.pi] * frames
        face_angles = [[a + v_mean, b + h_mean] for a, b in zip(face_pitchs, face_yaws)]
        fovs = [12] * frames
    elif setting == 2:   # CARLA
        h_mean = math.pi * (90 / 180)
        v_mean = math.pi * (60 / 180)
        generator.renderer.lock_view_dependence = False
        frames = opt.frames
        concat = None
        face_yaws = list(np.linspace(-np.pi, np.pi, frames + 1)[:-1])
        face_pitchs = [0*np.pi] * frames
        face_angles = [[a + v_mean, b + h_mean] for a, b in zip(face_pitchs, face_yaws)]
        fovs = [35] * frames

    seeds = opt.seeds
    generator.get_avg_w()

    for seed in tqdm(seeds):
        images = np.zeros((frames, config['global']['img_size'], config['global']['img_size'], 3), dtype=np.uint8)
        for i, ((pitch, yaw), fov) in enumerate(zip(face_angles, fovs)):
            config['camera']['fov'] = fov
            torch.manual_seed(seed)
            z = torch.randn((1, 256), device='cuda')
            with torch.no_grad():
                img = generator(z, **config['camera'], camera_origin=[np.sin(pitch) * np.cos(yaw), np.cos(pitch), np.sin(pitch) * np.sin(yaw)], truncation_psi=opt.psi)[0]
                img = img * 0.5 + 0.5
                img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                if not opt.no_watermark:
                    img = img*(1-watermark[...,-1:]) + watermark[...,-1:]*watermark[...,:-1]
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            images[i] = np.nan_to_num(img)
            if concat is None:
                imageio.imsave(os.path.join(opt.output_dir, f'grid_{seed}_{i}.png'),images[i])

        if concat is None:
            imageio.mimsave(os.path.join(opt.output_dir, f'grid_{seed}.gif'),images,fps=15)
        else:
            images = images.reshape((concat[0], concat[1], config['global']['img_size'], config['global']['img_size'], 3))
            images = np.concatenate(images, axis=-3)
            images = np.concatenate(images, axis=-2)
            imageio.imsave(os.path.join(opt.output_dir, f'grid_{seed}.png'), images)


