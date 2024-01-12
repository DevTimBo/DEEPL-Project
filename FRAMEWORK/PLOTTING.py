import numpy as np
import matplotlib.pyplot as plt
import cv2
def plot_n_images(images, titles, cmaps, figsize=(20, 5)):
    num_images = len(images)
    if num_images == 0 or num_images != len(titles) or num_images != len(cmaps):
        raise ValueError("Invalid number of images, titles, or cmaps provided.")
    if num_images == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(images[0], cmap=cmaps[0])
        ax.set_title(titles[0])
        ax.axis('off')
    else:
        fig, axs = plt.subplots(1, num_images, figsize=figsize)
        for i in range(num_images):
            rgb_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            print(f"image {i}")
            #axs[i].imshow(images[i], cmap=cmaps[i])
            axs[i].imshow(rgb_image)
            axs[i].set_title(titles[i])
            axs[i].axis('off')
    print("plit")
    plt.show()



def time_plot_with_avg(cmap, figsize, *images):
    num_images = len(images)

    if num_images == 0:
        print("No images provided.")
        return

    plt.figure(figsize=figsize)
    for i, image in enumerate(images, start=1):
        plt.subplot(1, num_images + 1, i)
        plt.imshow(image, cmap=cmap if len(image.shape) == 2 else None)
        plt.title(f"Time {i}")
        plt.axis('off')

    # Calculate and plot the average image
    avg_image = np.mean(images, axis=0)
    plt.subplot(1, num_images + 1, num_images + 1)
    plt.imshow(avg_image, cmap=cmap if len(avg_image.shape) == 2 else None)
    plt.title("Average Image")
    plt.axis('off')

    plt.show()


def plot_originals_and_noises(analyzed_images, noise_images, title):
    num_images = len(analyzed_images)

    if num_images == 0 or len(noise_images) != num_images:
        print("Invalid number of images provided.")
        return

    plt.figure(figsize=(15, 5 * num_images))
    plt.suptitle(title, fontsize=16)

    for i in range(num_images):
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(analyzed_images[i], cmap='viridis' if len(analyzed_images[i].shape) == 2 else None)
        plt.title(f"Analyzed Image {i + 1}")
        plt.axis('off')

        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(noise_images[i], cmap='viridis' if len(noise_images[i].shape) == 2 else None)
        plt.title(f"Noise Image {i + 1}")
        plt.axis('off')

    plt.show()


def overlap_and_plot(images, title="Overlapped Image"):
    num_images = len(images)

    if num_images == 0:
        print("No images provided.")
        return

    # Calculate the average image
    avg_image = np.mean(images, axis=0)

    # Plot the overlapped image
    plt.figure(figsize=(8, 8))

    plt.imshow(avg_image, cmap='viridis' if len(avg_image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')

    plt.show()
