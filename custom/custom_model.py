import cv2
import numpy as np


def read_mapping(csv_file_path):
    import csv

    class_dict = {}
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        header = next(csv_reader)
        num_classes = int(header[1])
        for row in csv_reader:
            class_number = int(row[0])
            class_name = row[1]
            class_dict[class_number] = class_name

    return class_dict, num_classes


csv_file_path = ''


def set_csv_file_path(csv_file_path_temp):
    global csv_file_path
    csv_file_path = csv_file_path_temp


size = (0, 0)


def set_size(size_temp):
    global size
    size = size_temp


channels = 1


def set_channels(channels_temp):
    global channels
    channels = channels_temp


def preprocess(img):
    if channels == 1:
        raise Exception("Not implemented yet")
    else:
        print(f"Size of image: {size}")
        print(f"Image shape: {img.shape}")
        resized_image = cv2.resize(img[0], size)
    print(resized_image.shape)
    resized_image = np.transpose(resized_image, [0, 1, 2])
    print(resized_image.shape)
    return resized_image[None]


def get_decode(csv_file_path, prediction, top=5):
    decoded_preds = []
    class_dict, num_classes = read_mapping(csv_file_path)

    # Assuming prediction is a list or array of probabilities
    if len(prediction[0]) == num_classes:
        # Enumerate through the probabilities and store (class_index, probability) tuples
        pred_tuples = [(i, pred_prob) for i, pred_prob in enumerate(prediction[0])]
        # Sort the tuples based on probability in descending order
        sorted_preds = sorted(pred_tuples, key=lambda x: x[1], reverse=True)

        # Take the top 'top' predictions
        top_preds = sorted_preds[:top]

        # Extract class labels and indices for the top predictions
        for class_index, pred_prob in top_preds:
            decoded_preds.append((class_index, class_dict[class_index],prediction[0][class_index]))
    print(decoded_preds)
    return decoded_preds


def decode_predictions(preds, top=5):
    result = get_decode(csv_file_path, preds, top)
    return [result]


if __name__ == "__main__":
    csv_file_path = 'mapping.csv'

    import keras
    import numpy as np


    model = keras.models.load_model('mnistRGB.keras')
    model.load_weights('mnistRGB_weights.keras')
    model.summary()
    image = cv2.imread("sample.png")
    image = cv2.resize(image, (28, 28))
    set_channels(3)
    set_size((28, 28))
    image = preprocess(image[None])
    prediction = model.predict(image)
    decoded_pred = decode_predictions(prediction)
    print(f"Prediction: {decoded_pred}")
    all_layers = model.layers
    for layer in reversed(all_layers):
        if 'conv' in layer.name:
            last_conv_layer = layer.name
            break
    img_size = (model.input_shape[1], model.input_shape[2])
    channels = model.input_shape[3]
