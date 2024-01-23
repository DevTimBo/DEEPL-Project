import numpy as np
def zero_weights(weights, percentage):
    modified_weights = []
    for w in weights:
        num_elements = int(percentage * w.size)
        zero_indices = np.random.choice(w.size, num_elements, replace=False)
        w_flat = w.flatten()
        w_flat[zero_indices] = 0
        w = w_flat.reshape(w.shape)

        modified_weights.append(w)

    return modified_weights


import keras.applications.vgg16 as vgg16
# Keras Model
keras_model = vgg16.VGG16(weights="imagenet")
keras_preprocess = vgg16.preprocess_input
keras_decode = vgg16.decode_predictions
last_conv_layer = "block5_conv3"
weights = keras_model.get_weights()


modified_weights = zero_weights(weights, 0.1)
import cv2

image = cv2.imread("tabby-cat.png")
image_processed = keras_preprocess(np.array(image)[None])
keras_model.set_weights(modified_weights)
prediction = keras_model.predict(image_processed)
print(keras_decode(prediction))

