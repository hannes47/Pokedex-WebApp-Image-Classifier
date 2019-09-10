from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import fastai
from fastai import *
from fastai.vision import *
import torch
fastai.basics.defaults.device = torch.device('cpu')

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with fastai
MODEL_PATH = os.path.join(os.getcwd(),'model')

cl_file = np.load('classes.npy')
classes = []
for n in range(len(cl_file)):
    classes.append(str(cl_file[n]))
data = ImageDataBunch.single_from_classes(MODEL_PATH, classes, ds_tfms=get_transforms(flip_vert=False),
                                          size = 128).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet101).load(os.path.join(os.getcwd(),'models','Pokemon_Resnet101_stage-3'))

# Load your trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, learn):
    #img = image.load_img(img_path, target_size=(224, 224))
    pred_class,_,losses = learn.predict(open_image(img_path))

#    # Preprocessing the image
#    x = image.img_to_array(img)
#    # x = np.true_divide(x, 255)
#    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
#    x = preprocess_input(x, mode='caffe')

#    preds = model.predict(x)
    return pred_class
#    return Flask.jsonify({
#            "predictions": sorted(
#                    zip(learn.data.classes, map(float,losses)),
#                    key=lambda p: p[1],
#                    reverse=True)})


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        basepath = os.getcwd()
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, learn)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = preds[0]
        result = str(preds)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
