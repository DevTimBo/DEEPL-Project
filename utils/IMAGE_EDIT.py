import numpy as np
import cv2

def add_noise(image, noise_level):
    np.random.seed(50)
    if noise_level > 255:
        noise_level = 255
    noise = np.random.normal(0, noise_level, image.shape)
    noise_image = image + noise
    noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
    return noise_image


def overlap_images(image_list):
    result_image = None
    print(len(image_list))
    for img in image_list:

        if result_image is None:
            # Initialize the result image with the first processed image
            result_image = img
        else:
            result_image = result_image + img

    result_image = result_image//len(image_list)
    print("Result image shape: ", result_image.shape)
    return result_image


