#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:49:27 2018

@author: jpatrickweller
"""

from flask import flash, request, redirect, url_for, render_template
from bb_style import app
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance
from werkzeug.utils import secure_filename
from keras.applications import VGG16
from keras.preprocessing import image as kimage 
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf


app.config.update(dict(
        UPLOAD_FOLDER = "./bb_style/static/uploads2/",
        DATA_FOLDER = "./bb_style/models",
        ))


#user = 'jpatrickweller' #add your username here (same as previous postgreSQL)            
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

#Load product names, prudct urls, and img urls
prod_data = pd.read_csv('bb_style/models/collection_data.csv')
prod_names = prod_data['product name']
prod_urls = prod_data['product url']
img_urls = prod_data['image url']
collection_features = np.load('bb_style/models/collection_features.npy')

# Define the model
model = VGG16(include_top=True, weights='imagenet')
#remove the classification layer (fc8)
model.layers.pop()
#remove the next fully connected layer (fc7)
model.layers.pop()
#fix the output of the model
model.outputs = [model.layers[-1].output]

# Defining a graph for some reason, see below
graph = tf.get_default_graph()

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/verB')
def verB():
    return render_template("indexB.html")


@app.route('/output', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            #return redirect(request.url)
            return render_template('index.html')
        file = request.files['file']
        #file = request.files.get('file')
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            
            #load image for processing through the model
            im = kimage.load_img(img_path, target_size=(224,224))
            im = kimage.img_to_array(im)
            im = np.expand_dims(im, axis=0)  
            
            # prepare the image for the VGG model
            im = preprocess_input(im)
        
            # pull out the feature matrix from (the 3rd to last layer of) the model
            # Attempting to fix a mysterious bug with a snipet of Adam's code
            global graph
            with graph.as_default():
                feat_mat = model.predict(im)
            
            # Save feature matrix as vector
            feature_vect = feat_mat.flatten()
             
            # Calculate distance of feature vector from pre-processed images
            nimages = len(collection_features)
            diffs = []
            for i in range(nimages):
                diffs.append(distance.cosine(feature_vect, collection_features[i,:]))
            diffs = 1 - np.array(diffs)
            
            # Store data in a pandas dataframe
            similar_images = pd.DataFrame({'product_name': prod_names,
                                'simscore': diffs,
                                'product_url': prod_urls,
                                'image_url': img_urls})
    
            # Reorganize by best matches
            similar_images.sort_values('simscore', ascending=False, inplace=True)
            
            # Return top match
            n_matches = sum(similar_images['simscore'] > .5)

            print("hello")
            print(similar_images.head())

            matches = similar_images.to_dict('records')
            matches = matches[:n_matches]

            imgpath2 = '/static/uploads2/Christening_Gown_Crochet.jpg'

            return render_template('output.html',
                                   n_matches=n_matches,
                                   matches=matches,
                                   original=imgpath2)
    
    return 'oh, farts. Something went wrong.'

'''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
