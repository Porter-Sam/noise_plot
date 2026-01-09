import numpy as np

class Trans:
    @staticmethod
    def tensor2img(img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img
    @staticmethod
    def single_tensor2img(img):
        img = np.round((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img