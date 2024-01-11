import numpy as np


def add_noise(image, noise_level):
    np.random.seed(50)
    noise = np.random.normal(0, noise_level, image.shape)
    noise_image = image + noise
    noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
    return noise_image
