#for pipeline
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

#main methode all others functions are called from there
def get_mcd_uncertainty(image, model, preprocess, decode, samples, dropoutRate, applyOrSkip, apply_skip_list):
    #load picture
    picture_input_preprocessed = preprocess(image[None])  
    
    #models
    modelNoMC = model
    modelMC = model

    #list of decoded predictions
    predictions_readable = []

    skip_apply_list = []

    #creating List once so not done in every iteration
    if (applyOrSkip == "Skip these layers"):
        skip_apply_list = skipLayers(apply_skip_list, modelMC)
    else:
        skip_apply_list = applyTo(apply_skip_list, modelMC)

    #apply MonteCarlo and write predictions
    for r in range(samples):    
        #skip or apply to these layers
        modelMC = applyMonteCarloApplyTo(skip_apply_list, modelMC, modelNoMC, dropoutRate)
        predictions = modelMC.predict(picture_input_preprocessed)
        predictions_readable.append(decode(predictions))
    
    #prediction    
    uncertainty = processPredictions(predictions_readable, modelNoMC, picture_input_preprocessed, decode)

    return uncertainty

def applyTo(applyToLayers, modelMC):
    #logic only apply to:
    applyTo = []

    #only apply to given layers
    for i in range(len(modelMC.layers)):
        if(modelMC.layers[i].__class__.__name__ in applyToLayers):
            applyTo.append(i)
    
    return applyTo

def skipLayers(skipTheseLayers, modelMC):
    #skip these layers   
    layerToSkip = []
    
    #these layers get skipped
    for i in range(len(modelMC.layers)):
        if(modelMC.layers[i].__class__.__name__ in skipTheseLayers):
            layerToSkip.append(i)

    return layerToSkip
    

def applyMonteCarloApplyTo(applytoList, modelMC, modelNoMC, dropoutRate):
    for i in range(len(modelMC.layers)):
        if((i) in applytoList):
            b = []
            b = modelNoMC.layers[i].get_weights()
            for q in range(len(b)):
                if(str(type(b[q])) == "<class 'numpy.float32'>"):
                    if(random.randint(1, 100) <= dropoutRate):
                        #print("1. change to zero")
                        b[q] = 0.00000000
                else:
                    for j in range(len(b[q])):
                        if(str(type(b[q][j])) == "<class 'numpy.float32'>"):
                            if(random.randint(1, 100) <= dropoutRate):
                                #print("2. change to zero")
                                b[q][j] = 0.00000000
                        else:
                            for k in range(len(b[q][j])):
                                if(str(type(b[q][j][k])) == "<class 'numpy.float32'>"):
                                    if(random.randint(1, 100) <= dropoutRate):
                                        #print("3. change to zero")
                                        b[q][j][k] = 0.00000000
                                else:
                                    for l in range(len(b[q][j][k])):
                                        if(str(type(b[q][j][k][l])) == "<class 'numpy.float32'>"):
                                            if(random.randint(1, 100) <= dropoutRate):
                                                #print("4. change to zero")
                                                b[q][j][k][l] = 0.00000000
                                        else:        
                                            for m in range(len(b[q][j][k][l])):
                                                if(str(type(b[q][j][k][l][m])) == "<class 'numpy.float32'>"):
                                                    if(random.randint(1, 100) <= dropoutRate):
                                                        #print("5. change to zero")
                                                        b[q][j][k][l][m] = 0.00000000
                                                else:
                                                    for n in range(len(b[q][j][k][l][m])):
                                                        if(str(type(b[q][j][k][l][m][n])) == "<class 'numpy.float32'>"):
                                                            if(random.randint(1, 100) <= dropoutRate):
                                                                #print("6. change to zero")
                                                                b[q][j][k][l][m][n] = 0.00000000
                                                        else:
                                                            for o in range(len(b[q][j][k][l][m][n])):
                                                                if(str(type(b[q][j][k][l][m][n][o])) == "<class 'numpy.float32'>"):
                                                                    if(random.randint(1, 100) <= dropoutRate):
                                                                        #print("7. change to zero")
                                                                        b[q][j][k][l][m][n][o] = 0.00000000
                modelMC.layers[i].set_weights(b)
    return modelMC

def processPredictions(predictions_readable, modelNoMC, picture_input_preprocessed, decode):
    counter = 0
    predictionNormalModel = decode(modelNoMC.predict(picture_input_preprocessed))
    for k in range(len(predictions_readable)):
        #print(predictions_readable[k])
        if (predictions_readable[k][0][0][1] == predictionNormalModel[0][0][1]):
            counter = counter + 1

    #print (predictionNormalModel)
    hits = str(counter/len(predictions_readable))
    return hits