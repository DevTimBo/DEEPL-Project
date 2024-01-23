import numpy as np
def zero_weights(weights, percentage):
    modified_weights = []
    for w in weights:
        print(w)
        num_elements = int(percentage * w.size)
        zero_indices = np.random.choice(w.size, num_elements, replace=False)
        w_flat = w.flatten()
        w_flat[zero_indices] = 0
        w = w_flat.reshape(w.shape)

        modified_weights.append(w)

    return modified_weights


def applyMonteCarloApplyTo(applytoList, modelMC, modelNoMC, dropoutRate):
    for i in range(len(modelMC.layers)):
        print(f"Layers Len: {len(modelMC.layers)}")
        if (i) in applytoList:
            print("Dense")
            b = modelNoMC.layers[i].get_weights()
            b = zero_weights(b, dropoutRate)

            modelMC.layers[i].set_weights(b)
    return modelMC

import keras.applications.vgg16 as vgg16
# Keras Model
keras_model = vgg16.VGG16(weights="imagenet")
keras_preprocess = vgg16.preprocess_input
keras_decode = vgg16.decode_predictions
last_conv_layer = "block5_conv3"
weights = keras_model.get_weights()



import cv2

image = cv2.imread("tabby-cat.png")
image_processed = keras_preprocess(np.array(image)[None])


keras_model = applyMonteCarloApplyTo([20,21,22], keras_model, keras_model, 0.3)

prediction = keras_model.predict(image_processed)
print(keras_decode(prediction))

