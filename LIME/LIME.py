from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

def get_lime_explanation(image, model, samples, features):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=5, hide_color=0,
                                             num_samples=samples)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=features,
                                                hide_rest=False)
    image = mark_boundaries(temp / 2 + 0.5, mask)
    return image

def get_lime_heat_map(image, model, samples):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=5, hide_color=0,
                                             num_samples=samples)
    # Select the same class explained on the figures above.
    ind =  explanation.top_labels[0]

    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    
    return heatmap 

    #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    #plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())

