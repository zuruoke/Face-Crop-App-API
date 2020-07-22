# Keras RetinaNet Face Crop API 

This Application involves code to build and run a Keras REST API that uses Keras RetinaNet to instinctively crop out scale and position-variant face(s) from pictures and save as a new image file.

To get more details how this scale and pose invariant Face Crop Model was trained, refer to this [repository](https://github.com/zuruoke/Face-Crop-App) 

# Getting Started

Use Google Colab for CUDA support with Cython.

## Installation

- Open up Google Colab and clone this repository.

      !git clone https://github.com/zuruoke/Face-Crop-App-API.git
      
- Navigate into the main directory and install all dependencies.

      %cd Face-Crop-App-API/
      !pip install .
      
- Run the code directly from the cloned repository to compile Cython code first

      !python setup.py build_ext --inplace
      
- Install flask-ngrok

      !pip install flask-ngrok
      
      
## Starting the Keras server

The Flask-Ngrok + Keras server can be started by running:
      
      !python run_server.py 
      2020-07-22 08:02:25.873928: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
      * Loading Zuruoke model and Flask starting server...please wait until server has fully started
      Using TensorFlow backend.
      2020-07-22 08:02:46.304407: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
      ..........
      * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
      * Running on http://c938e5256a37.ngrok.io
      * Traffic stats available on http://127.0.0.1:4040
      
You can now access the KERAS REST FLASK NGROK API via the ngrok URL (the URL with 'ngrok.io sufficed) 

We will use this url attached with an endpoint to send a POST request to the server.

# Consuming the Keras REST Flask-Ngrok API

We have to send a POST request to the server running to consume the API.

To consume the Keras REST API:

- First ensure *run_server.py* (i.e., the flask-ngrok web server) is currently running

- Then open another another Google Colab Notebook and clone this repo
   
      !git clone https://github.com/zuruoke/Face-Crop-App-API.git
   
- Navigate into the main directory, this time there's no need to install dependencies.

      %cd Face-Crop-App-API/
      
      
- Then run req.py - a new directory called **Face_Cropped_Images** will be created in the Face-Crop-App-API directory to store the Face Cropped Images

      !python req.py
      
- Next, You will be asked to enter the folder path to your query images and the Flask Ngrok API URL 
      
      Enter the Image Directory Path:/content/Query_Images/
      Enter the NGROK API URL:http://c938e5256a37.ngrok.io
      

- And then when you have provided the query image folder path and the KERAS FLASK NGROK API UR, All faces in the Images will be cropped and saved as a seperate image file.

- Check the Face-Crop-App-API/Face_Cropped_Images directory for the Face Cropped Images.





   
 **N/B: The Face Cropped Images are in BGR format, if you want it in RGB format, use Cv2 to convert to the RGB format

    
      


