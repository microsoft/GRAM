import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .representations.gram import *

from .renderers.manifold_renderer import *


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """Samples n random locations along a sphere of radius r. Uses a gaussian distribution for pitch and yaw"""
    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean
    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi # convert from radians to [0,1]
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)
    else:
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    camera_origin = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    camera_origin[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    camera_origin[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    camera_origin[:, 1:2] = r*torch.cos(phi)

    return camera_origin, torch.cat([phi, theta], dim=-1)


def get_camera_origins(camera_pos, r=1):
    n = camera_pos.shape[0]
    device = camera_pos.device
    phi = camera_pos[:, 0]
    theta = camera_pos[:, 1]
    camera_origin = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    camera_origin[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    camera_origin[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    camera_origin[:, 1:2] = r*torch.cos(phi)

    return camera_origin


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.step = 0


class GramGenerator(Generator):
    def __init__(self, z_dim, img_size, h_stddev ,v_stddev, h_mean, v_mean, sample_dist, representation_kwargs, renderer_kwargs, partial_grad=False):
        super().__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.h_stddev = h_stddev
        self.v_stddev = v_stddev
        self.h_mean = h_mean
        self.v_mean = v_mean
        self.sample_dist = sample_dist
        self.partial_grad = partial_grad
        self.representation = Gram(z_dim, **representation_kwargs)
        self.renderer = ManifoldRenderer(**renderer_kwargs)

    def _volume(self, z, truncation_psi=1):
        return lambda points, ray_directions: self.representation.get_radiance(z, points, ray_directions, truncation_psi)

    def _volume_with_frequencies_phase_shifts(self, freq, phase):
        return lambda points, ray_directions: self.representation.get_radiance_with_frequencies_phase_shifts(freq, phase, points, ray_directions)

    def _intersections(self, points, levels):
        return self.representation.get_intersections(points, levels)

    def get_avg_w(self):
        self.representation.get_avg_w()

    def forward_with_frequencies_phase_shifts(self, freq, phase, fov, ray_start, ray_end, img_size=None, camera_origin=None, camera_pos=None, patch=None):
        if camera_origin is None and camera_pos is None:
            camera_origin, camera_pos = sample_camera_positions(freq.device, freq.shape[0], 1, self.h_stddev, self.v_stddev, self.h_mean, self.v_mean, self.sample_dist)
        elif camera_origin is not None:
            camera_origin = torch.tensor(camera_origin, dtype=torch.float32, device=freq.device).reshape(1, 3).expand(freq.shape[0], 3)
        else:
            camera_origin = get_camera_origins(camera_pos)
        if img_size is None:
            img_size = self.img_size
        if patch is None:
            img, _ = self.renderer.render(self._intersections, self._volume_with_frequencies_phase_shifts(freq, phase), img_size, camera_origin, camera_pos, fov, ray_start, ray_end, freq.device, partial_grad = self.partial_grad)
        else:
            img = self.renderer.render_patch(self._intersections, self._volume_with_frequencies_phase_shifts(freq, phase), img_size, camera_origin, camera_pos, fov, ray_start, ray_end, freq.device, patch, partial_grad = self.partial_grad)
        return img, camera_pos

    def forward(self, z, fov, ray_start, ray_end, img_size=None, camera_origin=None, camera_pos=None, truncation_psi=1, patch=None):
        if camera_origin is None and camera_pos is None:
            camera_origin, camera_pos = sample_camera_positions(z.device, z.shape[0], 1, self.h_stddev, self.v_stddev, self.h_mean, self.v_mean, self.sample_dist)
        elif camera_origin is not None:
            camera_origin = torch.tensor(camera_origin, dtype=torch.float32, device=z.device).reshape(1, 3).expand(z.shape[0], 3)
        else:
            camera_origin = get_camera_origins(camera_pos)
        if img_size is None:
            img_size = self.img_size
        if patch is None:
            img, _ = self.renderer.render(self._intersections, self._volume(z, truncation_psi), img_size, camera_origin, camera_pos, fov, ray_start, ray_end, z.device)
        else:
            img = self.renderer.render_patch(self._intersections, self._volume(z, truncation_psi), img_size, camera_origin, camera_pos, fov, ray_start, ray_end, z.device, patch)
        return img, camera_pos

    @torch.no_grad()
    def experiment(self, z, fov, ray_start, ray_end, img_size=None, camera_origin=None, camera_pos=None, truncation_psi=1):
        if camera_origin is None and camera_pos is None:
            camera_origin, camera_pos = sample_camera_positions(z.device, z.shape[0], 1, self.h_stddev, self.v_stddev, self.h_mean, self.v_mean, self.sample_dist)
        elif camera_origin is not None:
            camera_origin = torch.tensor(camera_origin, dtype=torch.float32, device=z.device).reshape(1, 3).expand(z.shape[0], 3)
        else:
            camera_origin = get_camera_origins(camera_pos)
        if img_size is None:
            img_size = self.img_size
        img, exp_data = self.renderer.render(self._intersections, self._volume(z, truncation_psi), img_size, camera_origin, camera_pos, fov, ray_start, ray_end, z.device, detailed_output=True)
        return img, camera_pos, exp_data
