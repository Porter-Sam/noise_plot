import torch

def noise_mix(img, alpha, mu=0, sigma=1, schedule='VE'):
    noise = mu + sigma * torch.randn_like(img)
    if schedule == 'VE':
        return img + alpha * noise
    elif schedule == 'VP':
        return alpha * img + (1 - alpha) * noise