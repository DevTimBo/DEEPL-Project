# Autor: Jason Pranata
# Co-Autor: Tim Harmling
# Date: 13 February 2024 
# Description: This file contains the LIME functionsfor image classification that
# will be used in the main file to generate the LIME explanation in the framework

from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
from utils.IMAGE_EDIT import normalize_array_np

# Function to get LIME explanation
def get_lime_explanation(image, model, samples, features, preprocess, positive_only, hide_rest, mask_only, min_weight):
    # Preprocess image
    image = preprocess(np.array(image)[None])
    image = image[0]

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get LIME explanation
    explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0,
                                             num_samples=samples)
    
    # Get LIME explanation image
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=positive_only, num_features=features,
                                                hide_rest=hide_rest, min_weight=min_weight)
    
    # If mask_only is True, return only the mask
    if mask_only == True:
        image = mask
        
    # Else, return the image with the mask
    else:
        temp = temp[:, :, ::-1]  # Reversing the order of color channels from BGR back to RGB
        temp = temp / 2 + 0.5 # Adjust color brightness
        image = mark_boundaries(temp, mask)
        image = (image + image.max()).astype(np.float32)
    
    # Normalize the image
    image = normalize_array_np(image)
    
    return image

# Function to get LIME heat map
def get_lime_heat_map(image, model, samples, preprocess):
    # Preprocess image
    image = preprocess(np.array(image)[None])
    image = image[0]
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get LIME explanation
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
