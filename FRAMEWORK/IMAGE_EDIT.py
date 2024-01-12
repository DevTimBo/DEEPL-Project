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

        if result_image is None:
            # Initialize the result image with the first processed image
            result_image = img
        else:
            result_image = result_image + img

    result_image = result_image/len(image_list)
    return result_image.astype(np.uint8)


image = np.array([[5, 0, 0], [0, 50, 0], [0, 0, 255]])
image2 = np.array([[10, 0, 0], [0, 100, 0], [0, 0, 255]])
imagelist= []
imagelist.append(image)
imagelist.append(image2)

a = overlap_images(imagelist)
print(a)