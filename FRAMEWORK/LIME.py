from lime import lime_image
from skimage.segmentation import mark_boundaries




def get_lime_explanation(image, model, samples):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=5, hide_color=0,
                                             num_samples=samples)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    image = mark_boundaries(temp / 2 + 0.5, mask)
    return image
