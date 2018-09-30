#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:36:46 2018

@author: jpatrickweller
"""

from scipy.spatial import distance
import pandas as pd
import numpy as np
from keras.preprocessing import image as kimage 
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

def find_similar_imgs(pred_features, #features from user selected image
                 collection_features,  #list of features in the collection
                 prod_names, #list of filenames associated with the features
                 prod_urls, # list of urls associated with product filenames
                 img_urls, # list of urls associated with product images
                 ): 
    '''
    Finds matches for the features of the selected image, 
    according to the cosine distance metric specified.
    '''   
    pred_features = pred_features.flatten()
    nimages = len(collection_features)
    diffs = []
    for i in range(nimages):
        diffs.append(distance.cosine(pred_features, collection_features[i,:]))

    diffs = 1 - np.array(diffs)
    similar_images = pd.DataFrame({'product name': prod_names,
                                'simscore': diffs,
                                'product url': prod_urls,
                                'image url': img_urls})
    
    return(similar_images.sort_values('simscore', ascending=False))
    
    
def compute_feature_vect(image):
    
    # Images must be a standardized size (224 x 224)
    im = kimage.load_img(image, target_size=(224,224))
    
    # convert the image pixels to a numpy array
    im = kimage.img_to_array(image)
    
    # reshape data for the model
    im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
    
    # prepare the image for the VGG model
    im = preprocess_input(im)

    # Define the model
    # If I use the full model, I have to take off the last two layers to get features instead of identification
    model = VGG16(include_top=True, weights='imagenet')
    #remove the classification layer (fc8)
    model.layers.pop()
    #remove the next fully connected layer (fc7)
    model.layers.pop()
    #fix the output of the model
    model.outputs = [model.layers[-1].output]

    # pull out the feature matrix from (the 3rd to last layer of) the model
    feat_mat = model.predict(im)
    
    # Save feature matrix as vector
    feature_vect = feat_mat.flatten()
    return feature_vect
    

