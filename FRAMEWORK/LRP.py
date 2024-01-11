import innvestigate
import numpy as np
import PLOTTING

rule = "lrp.alpha_1_beta_0"


def analyze_image_lrp(image, model, preprocess):
    model = innvestigate.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(rule, model)
    image_processed = preprocess(np.array(image)[None])
    lrp_image = analyzer.analyze(image_processed)
    lrp_image = lrp_image.sum(axis=np.argmax(np.asarray(lrp_image.shape) == 3))
    lrp_image /= np.max(np.abs(lrp_image))
    lrp_image = lrp_image[0]
    return lrp_image


def lrp_simple(images):
    # Imports
    import keras.applications.vgg16 as vgg16
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    # Keras Model
    model = vgg16.VGG16(weights="imagenet")
    preprocess = vgg16.preprocess_input
    decode_predictions = vgg16.decode_predictions
    plot_images = []
    cmaps = []
    titles = []
    print(len(images))
    for image in images:
        plot_images.append(analyze_image_lrp(image, model, preprocess))
        titles.append("LRP")
        cmaps.append("viridis")
    PLOTTING.plot_n_images(plot_images, titles, cmaps, figsize=(5, 3))
