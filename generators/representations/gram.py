import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def geometry_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            m.weight.normal_(0,np.sqrt(2/num_output))
            nn.init.constant_(m.bias,0)


def geometry_init_last_layer(radius):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                nn.init.constant_(m.weight,10*np.sqrt(np.pi/num_input))
                nn.init.constant_(m.bias,-radius)
    return init


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=torch.sin):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return self.activation(freq * x + phase_shift)
    
    def statistic(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return self.activation(freq * x + phase_shift)


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_output_dim)
        )

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor


class GramSample(nn.Module):
    def __init__(self, hidden_dim_sample=64, layer_num_sample=3, center=(0,0,0), init_radius=0):
        super().__init__()
        self.hidden_dim = hidden_dim_sample
        self.layer_num = layer_num_sample

        self.network = [nn.Linear(3, self.hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(self.layer_num - 1):
            self.network += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True)]
        
        self.network = nn.Sequential(*self.network)

        self.output_layer = nn.Linear(self.hidden_dim, 1)

        self.network.apply(geometry_init)
        self.output_layer.apply(geometry_init_last_layer(init_radius))
        self.center = torch.tensor(center)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def calculate_intersection(self,intervals,vals,levels):
        intersections = []
        is_valid = []
        for interval,val,l in zip(intervals,vals,levels):
            x_l = interval[:,:,0]
            x_h = interval[:,:,1]
            s_l = val[:,:,0]
            s_h = val[:,:,1]
            scale = torch.where(torch.abs(s_h-s_l) > 0.05,s_h-s_l,torch.ones_like(s_h)*0.05)
            intersect = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*x_l + (l-s_l)*x_h)/scale,x_h)
            intersections.append(intersect)
            is_valid.append(((s_h-l<=0)*(l-s_l<=0)).to(intersect.dtype))
        
        return torch.stack(intersections,dim=-2),torch.stack(is_valid,dim=-2) #[batch,N_rays,level,3]
    
    def forward(self,input):
        x = input
        x = self.gridwarper(x)
        x = x - self.center.to(x.device)
        x = self.network(x)
        s = self.output_layer(x)

        return s

    def get_intersections(self, input, levels, **kwargs):
        # levels num_l
        batch,N_rays,N_points,_ = input.shape
        
        x = input.reshape(batch,-1,3)
        x = self.gridwarper(x)

        x = x - self.center.to(x.device)

        x = self.network(x)
        s = self.output_layer(x)

        s = s.reshape(batch,N_rays,N_points,1)
        s_l = s[:,:,:-1]
        s_h = s[:,:,1:]

        cost = torch.linspace(N_points-1,0,N_points-1).float().to(input.device).reshape(1,1,-1,1)
        x_interval = []
        s_interval = []
        for l in levels:
            r = (s_h-l <= 0) * (l-s_l <= 0) * 2 - 1
            r = r*cost
            _, indices = torch.max(r,dim=-2,keepdim=True)
            x_l_select = torch.gather(input,-2,indices.expand(-1, -1, -1, 3)) # [batch,N_rays,1]
            x_h_select = torch.gather(input,-2,indices.expand(-1, -1, -1, 3)+1) # [batch,N_rays,1]
            s_l_select = torch.gather(s_l,-2,indices)
            s_h_select = torch.gather(s_h,-2,indices)
            x_interval.append(torch.cat([x_l_select,x_h_select],dim=-2))
            s_interval.append(torch.cat([s_l_select,s_h_select],dim=-2))
        
        intersections,is_valid = self.calculate_intersection(x_interval,s_interval,levels)
        
        return intersections,s,is_valid


class GramRF(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=256, normalize=0.24, sigma_clamp_mode='softplus', rgb_clamp_mode='widen_sigmoid'):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.sigma_clamp_mode = sigma_clamp_mode
        self.rgb_clamp_mode = rgb_clamp_mode
        self.avg_frequencies = None
        self.avg_phase_shifts = None
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.color_layer = nn.ModuleList([FiLMLayer(hidden_dim + 3, hidden_dim)])

        self.output_sigma = nn.ModuleList([
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
        ])

        self.output_color = nn.ModuleList([
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
        ])
        
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + len(self.color_layer))*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.output_sigma.apply(frequency_init(25))
        self.color_layer.apply(frequency_init(25))
        self.output_color.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(normalize) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def get_avg_w(self):
        z = torch.randn((10000, self.z_dim), device=next(self.parameters()).device)
        with torch.no_grad():
            frequencies, phase_shifts = self.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts

    def forward(self, input, z, ray_directions, truncation_psi=1):
        frequencies, phase_shifts = self.mapping_network(z)
        if truncation_psi < 1:
            frequencies = self.avg_frequencies.lerp(frequencies, truncation_psi)
            phase_shifts = self.avg_phase_shifts.lerp(phase_shifts, truncation_psi)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, eps=1e-3):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
        sigma = 0
        rgb = 0
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            if index > 0:
                layer_sigma = self.output_sigma[index-1](x)
                if not index == 7:
                    layer_rgb_feature = x 
                else:
                    layer_rgb_feature = self.color_layer[0](torch.cat([ray_directions, x], dim=-1),\
                        frequencies[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim], phase_shifts[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim])
                layer_rgb = self.output_color[index-1](layer_rgb_feature)

                sigma += layer_sigma
                rgb += layer_rgb

        if self.rgb_clamp_mode == 'sigmoid':
            rgb = torch.sigmoid(rgb)
        elif self.rgb_clamp_mode == 'widen_sigmoid':
            rgb = torch.sigmoid(rgb)*(1+2*eps) - eps

        if self.sigma_clamp_mode == 'relu':
            sigma = F.relu(sigma)
        elif self.sigma_clamp_mode == 'softplus':
            sigma = F.softplus(sigma)

        return torch.cat([rgb, sigma], dim=-1)
        

class Gram(nn.Module):
    def __init__(self, z_dim=256, hidden_dim=256, normalize=0.24, sigma_clamp_mode='softplus', rgb_clamp_mode='widen_sigmoid', **sample_network_kwargs):
        super().__init__()
        self.sample_network = GramSample(**sample_network_kwargs)
        self.rf_network = GramRF(z_dim, hidden_dim, normalize, sigma_clamp_mode, rgb_clamp_mode)

    def get_avg_w(self):
        self.rf_network.get_avg_w()

    def get_intersections(self, points, levels):
        return self.sample_network.get_intersections(points, levels)

    def get_radiance(self, z, x, ray_directions, truncation_psi=1):
        return self.rf_network(x, z, ray_directions, truncation_psi)

    def get_radiance_with_frequencies_phase_shifts(self, frequencies, phase_shifts, x, ray_directions):
        return self.rf_network.forward_with_frequencies_phase_shifts(x, frequencies, phase_shifts, ray_directions)
