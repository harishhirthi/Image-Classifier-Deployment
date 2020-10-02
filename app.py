# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:44:33 2020

@author: Harish
"""
import os
from flask import Flask, flash, request, redirect, url_for, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app = Flask(__name__)
model = load_model('final_model.hdf5')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

#@app.route('/predict',methods=['POST'])
def predict(img):
    '''
    For rendering results on HTML GUI
    '''
    
    img  = load_img(img, target_size=(224,224))
    img = img_to_array(img)
    #img = np.array([img])
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pred = model.predict(img)
    ind = int(np.argmax(pred, axis = 1))

    return ind

@app.route('/result <ind>', methods=['GET', 'POST'])
def result(ind):
    if request.method == 'POST':
        return redirect(url_for('upload_file'))
    ind = int(ind)
    labels = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
    return render_template('index.html', prediction_text='Image is {}'.format(labels[ind]))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			l=predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('result',ind = l))
			#redirect(request.url)
	return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
