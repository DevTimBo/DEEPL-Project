import cv2


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


def get_decode(csv_file_path, prediction):
    decoded_pred = "No Prediction"
    class_dict, num_classes = read_mapping(csv_file_path)
    if len(prediction[0]) == num_classes:
        for i, pred_bit in enumerate(prediction[0]):
            if pred_bit >= 1:
                decoded_pred = class_dict[i]
    else:
        decoded_pred = class_dict[prediction[0]]
    return decoded_pred



if __name__ == "__main__":
    csv_file_path = 'mapping.csv'

    import keras
    import numpy as np
    size = (28, 28)
    model = keras.models.load_model('mnist_int.keras')
    model.load_weights('mnist_weights_int.keras')

    image = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)

    resize_image = cv2.resize(image, size)
    image_processed = np.array(resize_image)[None]
    prediction = model.predict(image_processed)
    decoded_pred = get_decode(csv_file_path, prediction)
    print(f"Prediction: {decoded_pred}")





