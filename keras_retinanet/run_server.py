import tensorflow as tf
import numpy as np
import requests
import urllib
import os
import pandas as pd
from PIL import Image
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import flask


app = flask.Flask(__name__)
model = None


def load_model():
	 urllib.request.urlretrieve("https://github.com/zuruoke/Face-Crop-App-API/releases/download/v0.1-alpha/crop_weight.h5", "FACE_CROP.h5")

	 global model
	 model = "FACE_CROP.h5"	

def prepare_image(image):
	image = image[:,:,:3]
    imp = preprocess_image(image)
    imp, scale = resize_image(image)

    return imp, scale


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	

	if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            IMG = flask.request.files["image"]
            imp = flask.request.files["image"].read()
            imp = Image.open(io.BytesIO(imp))

            # preprocess the image and prepare it for classification
            imp, scale = prepare_image(imp)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(imp, axis=0))
            boxes /= scale
            data["predictions"] = []
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < 0.5:
                    break
                box = box.astype(np.int32)

                r = {"image_name":IMG, "x_min": box[1], "y_min": box[3], "x_max": box[0], "y_max": box[2]}
            	data["predictions"].append(r)

            	data["success"] = True	

    return flask.jsonify(data)   



if __name__ == "__main__":
    print(("* Loading Zuruoke model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(port=8000, debug=True)           