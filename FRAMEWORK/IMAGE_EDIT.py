import numpy as np
import cv2

def add_noise(image, noise_level):
    np.random.seed(50)
    noise = np.random.normal(0, noise_level, image.shape)
    noise_image = image + noise
    noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
    return noise_image


def overlap_images(image_list):
    result_image = None
    print(len(image_list))
    for img in image_list:
        # Check the number of channels in the image
        num_channels = img.shape[-1] if len(img.shape) == 3 else 1

        if num_channels == 1:
            # Convert grayscale to 3-channel image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif num_channels == 3:
            img_rgb = img
        else:
            print("Unsupported number of channels. Must be 1 or 3.")

        if result_image is None:
            # Initialize the result image with the first processed image
            result_image = img_rgb
        else:
            result_image = (result_image + img_rgb)/2
    return result_image
