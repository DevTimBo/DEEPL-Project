from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
from utils.IMAGE_EDIT import normalize_array_np

def get_lime_explanation(image, model, samples, features, preprocess, positive_only, hide_rest):
    image = preprocess(np.array(image)[None])
    image = image[0]
    print(f"Image shape: {image.shape}")
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0,
                                             num_samples=samples)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=positive_only, num_features=features,
                                                hide_rest=hide_rest)
    image = mark_boundaries(temp / 2 + 0.5, mask)
    image = (image + image.max()).astype(np.float32)
    image = normalize_array_np(image)
    return image


def get_lime_heat_map(image, model, samples, preprocess):
    image = preprocess(np.array(image)[None])
    image = image[0]
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0,
                                             num_samples=samples)
    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]

    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    heatmap = (heatmap + heatmap.max()).astype(np.float32)
    heatmap = normalize_array_np(heatmap)
    print(f"LIME {heatmap.max()}")
    return heatmap


