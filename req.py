# -*- coding: utf-8 -*-
"""req.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Du3XCl-h2i4jSzUou39rwBAgp7pJ7oob
"""

import os
os.mkdir("Reed")

import requests
import json
import ast
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
# initialize the Keras REST API endpoint URL along with the input
# image path
os.mkdir("Snake")
IMG_PATH = input("Enter the Image Directory Path:")
NGROK_API = input("Enter the NGROK API URL:")
Pred = []

IMAGES = os.listdir(IMG_PATH)
for IM in IMAGES:
  IMAGE_PATH = os.path.join(IMG_PATH, IM)
  KERAS_REST_API_URL = NGROK_API+"/predict"
  image = open(IMAGE_PATH, "rb").read()
  payload = {"image": image}

# submit the request
  r = requests.post(KERAS_REST_API_URL, files=payload).json()
  y = ast.literal_eval(r)
  for n in y["predictions"]:
    n["img_dir"] = IMAGE_PATH
  Pred.append(y)

x = pd.DataFrame(Pred)

converted_data_train = {
    'img_dir': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': []
}

def map_to_data(row, converted_data):


  # there could be more than 1 face per image
  for anno in row['predictions']:
    filepath = anno['img_dir']
    converted_data['img_dir'].append(filepath)

    x_min = anno['x_min']
    converted_data['x_min'].append(x_min)

    x_max = anno['x_max']
    converted_data['x_max'].append(x_max)

    y_min = anno['y_min']
    converted_data['y_min'].append(y_min)

    y_max = anno['y_max']
    converted_data['y_max'].append(y_max)

x.apply(lambda row: map_to_data(row, converted_data_train), axis=1)

df = pd.DataFrame(converted_data_train)

path_2 = 'Snake/'
idl = 0
def save_cropped_image(_df):
    global idl
    for i,r in _df.iterrows():
        n =  r["img_dir"]
        x1 = r["x_min"]
        x2 = r["y_min"]
        y1 = r["x_max"]
        y2 = r["y_max"]
        im = np.array(Image.open(n))
        new = im[x1:x2, y1:y2]
        f = n.split("/")[-1]
        cv2.imwrite("{}{}_{}".format(path_2,idl,f), new)
        idl +=1

save_cropped_image(df)