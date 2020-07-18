# USAGE
# python simple_request.py

# import the necessary packages
import requests
from flask_ngrok import run_with_ngrok
# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://30c3b91013f8.ngrok.io/"
IMAGE_PATH = "gart.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	# loop over the predictions and display them
	for (i, result) in enumerate(r["predictions"]):
		print("{}. {}: {:.4f}".format(i + 1, result["x_min"],
			result["x_max"]))

# otherwise, the request failed
#else:
	#print("Request failed")