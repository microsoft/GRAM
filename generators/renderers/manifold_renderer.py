import math
import numpy as np
import torch
import torch.nn.functional as F

from .math_utils_torch import *


def fancy_integration(rgb_sigma, z_vals, device, is_valid, bg_pos, last_back=False, white_back=False, delta_alpha=0.04, delta_final=1e10):
    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    deltas = torch.ones_like(z_vals[:, :, 1:] - z_vals[:, :, :-1])*delta_alpha
    delta_inf = delta_final * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2) # [batch,N_rays,num_manifolds,1]

    bg_pos = F.one_hot(bg_pos.squeeze(-1),num_classes=deltas.shape[-2]).to(torch.bool) # [batch,N_rays,num_manifolds]
    bg_pos = bg_pos.unsqueeze(-1) # [batch,N_rays,num_manifolds,1]
    deltas[bg_pos] = delta_final

    alphas = 1-torch.exp(-deltas * sigmas)
    alphas = alphas*is_valid
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    T = torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights = alphas * T
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)/weights_sum

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    return rgb_final, depth_final, weights, T


def get_initial_rays_trig(n, num_samples, device, fov, resolution, ray_start, ray_end, randomize=True, patch_range=None):
    """Returns sample points, z_vals, ray directions in camera space."""
    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    if patch_range is not None:
        x = x[patch_range[0]:patch_range[1]]
        y = y[patch_range[0]:patch_range[1]]
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    z_vals = torch.linspace(ray_start, ray_end, num_samples, device=device).reshape(1, num_samples, 1).repeat(rays_d_cam.shape[0], 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_samples, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)
    
    if randomize:
        perturb_points(points, z_vals, rays_d_cam, device)

    return points, z_vals, rays_d_cam


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def get_intersection_with_MPI(transformed_ray_directions,transformed_ray_origins,device, mpi_start=0.12,mpi_end=-0.12,mpi_num=24):
    mpi_z_vals = torch.linspace(mpi_start, mpi_end, mpi_num, device=device)
    z_vals = mpi_z_vals.view(1,1,mpi_num) - transformed_ray_origins[...,-1:] #[batch,N,mpi_num]
    z_vals = z_vals/transformed_ray_directions[...,-1:] #[batch,N,mpi_num]
    z_vals = z_vals.unsqueeze(-1)
    points = transformed_ray_origins.unsqueeze(2) + transformed_ray_directions.unsqueeze(2)*z_vals

    return points, z_vals


def transform_sampled_points(points, ray_directions, camera_origin, camera_pos, device):
    n, num_rays, num_samples, channels = points.shape
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_samples x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_samples, 4)
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], transformed_ray_directions, transformed_ray_origins, camera_pos


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


class ManifoldRenderer:
    def __init__(self, num_manifolds, levels_start, levels_end, num_samples, last_back=False, white_back=False, background=True, delta_alpha=0.04, delta_final=1e10, lock_view_dependence=False) -> None:
        self.num_manifolds = num_manifolds
        self.levels_start = levels_start
        self.levels_end = levels_end
        self.num_samples = num_samples
        self.last_back = last_back
        self.white_back = white_back
        self.background = background
        self.delta_alpha = delta_alpha
        self.delta_final = delta_final if background else delta_alpha
        self.lock_view_dependence = lock_view_dependence
    
    def render(self, intersection, volume, img_size, camera_origin, camera_pos, fov, ray_start, ray_end, device, detailed_output=False, partial_grad=False):
        batchsize = camera_origin.shape[0]

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batchsize, self.num_samples, resolution=(img_size, img_size), device=device, fov=fov, ray_start=ray_start, ray_end=ray_end, randomize=False) # batchsize, pixels, num_manifolds, 1
            transformed_points_sample, transformed_ray_directions, transformed_ray_origins, _ = transform_sampled_points(points_cam, rays_d_cam, camera_origin, camera_pos, device=device)
            transformed_points_sample = transformed_points_sample.reshape(batchsize, img_size*img_size, -1, 3)
            if self.background:
                levels = torch.linspace(self.levels_start, self.levels_end, self.num_manifolds-1).to(device)
                transformed_points_back, _ = get_intersection_with_MPI(transformed_ray_directions,transformed_ray_origins,device=device,mpi_start=-0.12,mpi_end=-0.12,mpi_num=1)
            else:
                levels = torch.linspace(self.levels_start, self.levels_end, self.num_manifolds).to(device)
        
        if not partial_grad:
            transformed_points,_,is_valid = intersection(transformed_points_sample, levels) # [batch,H*W,num_manifolds,3]
        else:
            with torch.no_grad():
                transformed_points,_,is_valid = intersection(transformed_points_sample, levels) # [batch,H*W,num_manifolds,3]

        if self.background:
            transformed_points = torch.cat([transformed_points,transformed_points_back],dim=-2)
            is_valid = torch.cat([is_valid,torch.ones(is_valid.shape[0],is_valid.shape[1],1,is_valid.shape[-1]).to(is_valid.device)],dim=-2)

        with torch.no_grad():
            z_vals = torch.sqrt(torch.sum((transformed_points - transformed_ray_origins.unsqueeze(2))**2,dim=-1,keepdim=True)) # [batch,H*W,num_manifolds,1]

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, self.num_manifolds, -1)   

            if self.lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
        
        
        if not partial_grad:
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batchsize, img_size*img_size*self.num_manifolds, 3)
            transformed_points = transformed_points.reshape(batchsize, img_size*img_size*self.num_manifolds, 3)
            coarse_output = volume(transformed_points, transformed_ray_directions_expanded).reshape(batchsize, img_size * img_size, self.num_manifolds, 4)
        else:
            indices_pts = torch.randperm(img_size*img_size)
            indices_with_grad = indices_pts[:img_size*img_size//4]
            indices_wo_grad = indices_pts[img_size*img_size//4:]    

            transformed_ray_directions_expanded_with_grad = transformed_ray_directions_expanded[:,indices_with_grad]
            transformed_ray_directions_expanded_wo_grad = transformed_ray_directions_expanded[:,indices_wo_grad]

            transformed_points_with_grad = transformed_points[:,indices_with_grad]
            transformed_points_wo_grad = transformed_points[:,indices_wo_grad]   
            transformed_points_with_grad = transformed_points_with_grad.reshape(batchsize, -1, 3)
            transformed_ray_directions_expanded_with_grad = transformed_ray_directions_expanded_with_grad.reshape(batchsize, -1, 3)  

            coarse_output_with_grad = volume(transformed_points_with_grad, transformed_ray_directions_expanded_with_grad).reshape(batchsize, -1, self.num_manifolds, 4)   
            with torch.no_grad():
                transformed_points_wo_grad = transformed_points_wo_grad.reshape(batchsize, -1, 3)
                transformed_ray_directions_expanded_wo_grad = transformed_ray_directions_expanded_wo_grad.reshape(batchsize, -1, 3)
                coarse_output_wo_grad = volume(transformed_points_wo_grad, transformed_ray_directions_expanded_wo_grad).reshape(batchsize, -1, self.num_manifolds, 4)
        
            coarse_output = torch.zeros(batchsize, img_size*img_size, self.num_manifolds, 4).to(coarse_output_with_grad.device).to(coarse_output_with_grad.dtype)
            coarse_output[:,indices_with_grad] = coarse_output_with_grad
            coarse_output[:,indices_wo_grad] = coarse_output_wo_grad

        all_points = transformed_points.reshape(batchsize, img_size * img_size, self.num_manifolds, 3)
        all_outputs = coarse_output
        all_z_vals = z_vals

        _, indices = torch.sort(all_z_vals, dim=-2)
        all_z_vals = torch.gather(all_z_vals, -2, indices)
        all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        all_points = torch.gather(all_points, -2, indices.expand(-1, -1, -1, 3))
        is_valid = torch.gather(is_valid,-2,indices)

        bg_pos = torch.argmax(indices,dim=-2)

        pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, is_valid=is_valid, bg_pos=bg_pos, device=device, white_back=self.white_back, last_back=self.last_back, delta_final=self.delta_final, delta_alpha=self.delta_alpha)

        pixels = pixels.reshape((batchsize, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        if detailed_output:
            detail = {
                'points': all_points.reshape(batchsize, img_size, img_size, -1, 3),
                'outputs': all_outputs.reshape(batchsize, img_size, img_size, -1, 4),
                'z_vals': all_z_vals.reshape(batchsize, img_size, img_size, -1, 1),
                'depth': depth.reshape(batchsize, img_size, img_size, -1, 1),
                'weights': weights.reshape(batchsize, img_size, img_size, -1, 1),
            }
        else:
            detail = None

        return pixels, detail

    def render_patch(self, intersection, volume, img_size, camera_origin, camera_pos, fov, ray_start, ray_end, device, patch, partial_grad=False):
        batchsize = camera_origin.shape[0]
        patch_idx = patch[0]
        patch_num = patch[1]
        patch_range = (img_size*img_size*patch_idx//patch_num, img_size*img_size*(patch_idx+1)//patch_num)
        patch_len = patch_range[1] - patch_range[0]

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batchsize, self.num_samples, resolution=(img_size, img_size), device=device, fov=fov, ray_start=ray_start, ray_end=ray_end, randomize=False, patch_range=patch_range) # batchsize, pixels, num_manifolds, 1
            transformed_points_sample, transformed_ray_directions, transformed_ray_origins, _ = transform_sampled_points(points_cam, rays_d_cam, camera_origin, camera_pos, device=device)
            transformed_points_sample = transformed_points_sample.reshape(batchsize, patch_len, -1, 3)
            if self.background:
                levels = torch.linspace(self.levels_start, self.levels_end, self.num_manifolds-1).to(device)
                transformed_points_back, _ = get_intersection_with_MPI(transformed_ray_directions,transformed_ray_origins,device=device,mpi_start=-0.12,mpi_end=-0.12,mpi_num=1)
            else:
                levels = torch.linspace(self.levels_start, self.levels_end, self.num_manifolds).to(device)

        if not partial_grad:
            transformed_points,_,is_valid = intersection(transformed_points_sample, levels) # [batch,H*W,num_manifolds,3]
        else:
            with torch.no_grad():
                transformed_points,_,is_valid = intersection(transformed_points_sample, levels) # [batch,H*W,num_manifolds,3]

        if self.background:
            transformed_points = torch.cat([transformed_points,transformed_points_back],dim=-2)
            is_valid = torch.cat([is_valid,torch.ones(is_valid.shape[0],is_valid.shape[1],1,is_valid.shape[-1]).to(is_valid.device)],dim=-2)

        with torch.no_grad():
            z_vals = torch.sqrt(torch.sum((transformed_points - transformed_ray_origins.unsqueeze(2))**2,dim=-1,keepdim=True)) # [batch,patch_len,num_manifolds,1]

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, self.num_manifolds, -1)   

            if self.lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        if not partial_grad:
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batchsize, patch_len*self.num_manifolds, 3)
            transformed_points = transformed_points.reshape(batchsize, patch_len*self.num_manifolds, 3)
            coarse_output = volume(transformed_points, transformed_ray_directions_expanded).reshape(batchsize, patch_len, self.num_manifolds, 4)
        else:
            indices_pts = torch.randperm(patch_len)
            indices_with_grad = indices_pts[:patch_len//4]
            indices_wo_grad = indices_pts[patch_len//4:]    

            transformed_ray_directions_expanded_with_grad = transformed_ray_directions_expanded[:,indices_with_grad]
            transformed_ray_directions_expanded_wo_grad = transformed_ray_directions_expanded[:,indices_wo_grad]

            transformed_points_with_grad = transformed_points[:,indices_with_grad]
            transformed_points_wo_grad = transformed_points[:,indices_wo_grad]   
            transformed_points_with_grad = transformed_points_with_grad.reshape(batchsize, -1, 3)
            transformed_ray_directions_expanded_with_grad = transformed_ray_directions_expanded_with_grad.reshape(batchsize, -1, 3)  

            coarse_output_with_grad = volume(transformed_points_with_grad, transformed_ray_directions_expanded_with_grad).reshape(batchsize, -1, self.num_manifolds, 4)   
            with torch.no_grad():
                transformed_points_wo_grad = transformed_points_wo_grad.reshape(batchsize, -1, 3)
                transformed_ray_directions_expanded_wo_grad = transformed_ray_directions_expanded_wo_grad.reshape(batchsize, -1, 3)
                coarse_output_wo_grad = volume(transformed_points_wo_grad, transformed_ray_directions_expanded_wo_grad).reshape(batchsize, -1, self.num_manifolds, 4)
        
            coarse_output = torch.zeros(batchsize, patch_len, self.num_manifolds, 4).to(coarse_output_with_grad.device).to(coarse_output_with_grad.dtype)
            coarse_output[:,indices_with_grad] = coarse_output_with_grad
            coarse_output[:,indices_wo_grad] = coarse_output_wo_grad        

        all_points = transformed_points.reshape(batchsize, patch_len, self.num_manifolds, 3)
        all_outputs = coarse_output
        all_z_vals = z_vals

        _, indices = torch.sort(all_z_vals, dim=-2)
        all_z_vals = torch.gather(all_z_vals, -2, indices)
        all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        all_points = torch.gather(all_points, -2, indices.expand(-1, -1, -1, 3))
        is_valid = torch.gather(is_valid,-2,indices)

        bg_pos = torch.argmax(indices,dim=-2)

        pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, is_valid=is_valid, bg_pos=bg_pos, device=device, white_back=self.white_back, last_back=self.last_back, delta_final=self.delta_final, delta_alpha=self.delta_alpha)

        pixels = pixels.reshape((batchsize, patch_len, 3)).permute(0, 2, 1) * 2 - 1

        return pixels
