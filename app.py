# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:22:41 2020

@author: Lenovo
"""

from __future__ import division,print_function
import sys
import os
import glob
import re
import numpy as np

#Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Flask Utils
from flask import Flask, redirect, url_for,request,render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)

MODEL_PATH='model_vgg19.h5'

#load your trained model
model=load_model(MODEL_PATH)

def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x,axis=0)
    
    x=preprocess_input(x)
    preds=model.predict(x)
    preds=np.argmax(preds,axis=1)
    if preds==0:
        preds="The Person is Infected  with Pneumonia "
    else:
        preds="The person is not Infected with Pneumonia"
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
                basepath,'uploads',secure_filename(f.filename)
                )
        f.save(file_path)
        
        preds=model_predict(file_path,model)
        results=preds
        return results
    return None

if __name__=='__main__':
    app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    