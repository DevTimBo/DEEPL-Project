

import innvestigate
import numpy as np
import PLOTTING
import tensorflow as tf
import matplotlib.pyplot as plt

#tf.compat.v1.disable_eager_execution()
#rule = "lrp.alpha_1_beta_0"



def analyze_image_lrp(image, model, preprocess, rule):
    model = innvestigate.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(rule, model)
    image_processed = preprocess(np.array(image)[None])
    lrp_image = analyzer.analyze(image_processed)
    lrp_image = lrp_image.sum(axis=np.argmax(np.asarray(lrp_image.shape) == 3))
    lrp_image /= np.max(np.abs(lrp_image))
    lrp_image = lrp_image[0]
    return lrp_image
