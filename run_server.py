import tensorflow as tf
import numpy as np
import requests
import urllib
import os
import pandas as pd
from PIL import Image
from keras_retinanet import models
from tensorflow.keras.preprocessing.image import img_to_array
from keras_retinanet.utils.image import preprocess_image, resize_image
import flask
import io
import uuid
from flask_ngrok import run_with_ngrok



app = flask.Flask(__name__)
run_with_ngrok(app)
model = None


def load_model():
    urllib.request.urlretrieve("https://github.com/zuruoke/Face-Crop-App-API/releases/download/v0.1-alpha/crop_weight.h5", "FACE_CROP.h5")
    global model
    model_file = 'FACE_CROP.h5'
    model = models.load_model(model_file)
    model = models.convert_model(model)
    

def prepare_image(image):   
    if image.mode != "RGB":     
        image = image.convert("RGB")
    image = img_to_array(image)       
    image = preprocess_image(image)   
    image, scale = resize_image(image)    
    return image, scale


@app.route("/predict", methods=["POST"])
def predict():
	

    data = {"success": False}      
    if flask.request.method == "POST":  
        if flask.request.files.get("image"):
            # read the image in PIL format   
            image = flask.request.files["image"].read()   
            image = Image.open(io.BytesIO(image))
            k = str(uuid.uuid4())

            # preprocess the image and prepare it for classification    
            image, scale = prepare_image(image) 
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale  
            data["predictions"] = []    
            for box, score, label in zip(boxes[0], scores[0], labels[0]):   
                if score < 0.5: 
                    break   
                    box = box.astype(np.int32)  
                r = {"image_name":k, "x_min": box[1], "y_min": box[3], "x_max": box[0], "y_max": box[2]}  
                data["predictions"].append(r)   
                data["success"] = True      
    return flask.jsonify(str(data))   



if __name__ == "__main__":
    print(("* Loading Zuruoke model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()           
