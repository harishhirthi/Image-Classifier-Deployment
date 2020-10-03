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

    return pred[0], ind

@app.route('/result <prob0> <prob1> <prob2> <prob3> <prob4> <prob5> <prob6> <prob7> <ind>', methods=['GET', 'POST'])
def result(prob0, prob1, prob2, prob3, prob4, prob5, prob6, prob7, ind):
    if request.method == 'POST':
        return redirect(url_for('upload_file'))
    ind = int(ind)
    labels = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
    prob0 = round(float(prob0), 3) * 100
    prob1 = round(float(prob1), 3) * 100
    prob2 = round(float(prob2), 3) * 100
    prob3 = round(float(prob3), 3) * 100
    prob4 = round(float(prob4), 3) * 100
    prob5 = round(float(prob5), 3) * 100
    prob6 = round(float(prob6), 3) * 100
    prob7 = round(float(prob7), 3) * 100
    return render_template('index.html', air=prob0, car=prob1, cat=prob2, dog=prob3, flower=prob4, fruit=prob5, motor=prob6,
                           person=prob7, prediction_text='Image is {}'.format(labels[ind]))

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
			l, index=predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('result', prob0=l[0], prob1=l[1], prob2=l[2], prob3=l[3], prob4=l[4], prob5=l[5], prob6=l[6], prob7=l[7], ind=index))
			#redirect(request.url)
	return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
