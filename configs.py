import math

FFHQ_default = {
    'global': {
        'img_size': 256,
        'batch_size': 4, # batchsize per GPU. We use 8 GPUs by default so that the effective batchsize for an iteration is 4*8=32
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 1.,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 256,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
                'last_back': False,
                'white_back': False,
                'background': True,
            }
        }
    },
    'discriminator': {
        'class': 'GramDiscriminator',
        'kwargs': {
            'img_size': 256,
        }
    },
    'dataset': {
        'class': 'FFHQ',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

CATS_default = {
    'global': {
        'img_size': 256,
        'batch_size': 2,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 1.,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 2,
            'real_pos_lambda': 30.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 256,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 64,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
                'last_back': False,
                'white_back': False,
                'background': True,
            }
        }
    },
    'discriminator': {
        'class': 'GramDiscriminator',
        'kwargs': {
            'img_size': 256,
        }
    },
    'dataset': {
        'class': 'CATS',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

CARLA_default = {
    'global': {
        'img_size': 128,
        'batch_size': 4,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'sampling_network_lr': 2e-6,
        'betas': (0, 0.9),
        'grad_clip': 0.3,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 128,
            'h_stddev': math.pi,
            'v_stddev': math.pi*(42.5/180),
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*(42.5/180),
            'sample_dist': 'spherical_uniform',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'normalize': 2,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 256,
                'layer_num_sample': 3,
                'center': (0, 0, 0),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 48,
                'levels_start': 35,
                'levels_end': 5,
                'delta_alpha': 0.02,
                'last_back': False,
                'white_back': True,
                'background': False,
            }
        }
    },
    'discriminator': {
        'class': 'GramDiscriminator',
        'kwargs': {
            'img_size': 128,
        }
    },
    'dataset': {
        'class': 'CARLA',
        'kwargs': {
            'img_size': 128,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 30,
        'ray_start': 0.7,
        'ray_end': 1.3,
    }
}
