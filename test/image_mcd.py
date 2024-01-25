import cv2
import numpy as np
def create_mcd_image( size, text1, text2, text3):

    from PIL import Image, ImageDraw, ImageFont
    # Set image dimensions
    width, height = size

    # Create a white background image
    image = Image.new("RGB", (width, height), "white")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Set font properties
    font_size = 20
    font = ImageFont.truetype("arial.ttf", font_size)  # Use a suitable font file path

    # Set text positions
    x1, y1 = int(width * 0.4), int(height * 0.2)
    x2, y2 = int(width * 0.4), int(height * 0.4)
    x3, y3 = int(width * 0.4), int(height * 0.6)

    # Draw black text on the image
    draw.text((x1, y1), text1, font=font, fill="black")
    draw.text((x2, y2), text2, font=font, fill="black")
    draw.text((x3, y3), text3, font=font, fill="black")

    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)
    image_array = convert_to_uint8(image_array)
    return image_array


def convert_to_uint8(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Scale float values to the range [0, 255] and convert to uint8
        return (image * 255).clip(0, 255).astype(np.uint8)
    elif image.dtype == np.uint8:
        # Image is already uint8, no need to convert
        return image
    else:
        # Handle other data types or raise an error if needed
        raise ValueError("Unsupported data type. Supported types are float32, float64, and uint8.")
import matplotlib.pyplot as plt
def plot_n_images(images, titles, cmaps, max_images_per_row, figsize=(20, 5)):
    num_images = len(images)
    if num_images == 0 or num_images != len(titles) or num_images != len(cmaps):
        raise ValueError("Invalid number of images, titles, or cmaps provided.")
    if num_images == 1:
        num_row = 1

        fig, ax = plt.subplots(num_row, 1, figsize=figsize)
        ax.imshow(images[0], cmap=cmaps[0])
        ax.set_title(titles[0])
        ax.axis('off')

    else:
        num_rows = 1
        if num_images > max_images_per_row:
            num_rows = (num_images - 1) // max_images_per_row + 1

        fig, axs = plt.subplots(num_rows, max_images_per_row, figsize=figsize)
        print(f"num rows {num_rows}")
        print(f"max_images_per_row {max_images_per_row}")
        if max_images_per_row == 1 or num_rows == 1:
            for i in range(num_images):
                rgb_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                print("imshow")
                axs[i].imshow(rgb_image)
                print("titles")
                axs[i].set_title(titles[i])
                print("off")
                axs[i].axis('off')

        else:
            for i in range(num_images):
                print(i)
                row_index = i // max_images_per_row
                col_index = i % max_images_per_row
                print(row_index, col_index)
                print("RGB")
                rgb_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                print("imshow")
                axs[row_index, col_index].imshow(rgb_image)
                print("titles")
                axs[row_index, col_index].set_title(titles[i])
                print("off")
                axs[row_index, col_index].axis('off')
    print("Plot")
    plt.show()


# Example usage
image_size = (224, 224)
text1 = "Hello"
text2 = "World"
text3 = "Python"

resulting_image = create_mcd_image(image_size, text1, text2, text3)
images = [resulting_image]
titles = ["Image 1"]
cmaps = ["gray"]
max_imagtes = 1

plot_n_images(images, titles, cmaps, max_imagtes)
