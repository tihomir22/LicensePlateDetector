import os
from ultralytics import YOLO
from flask import Flask
import yaml
from predictFn import PredictLicensePlate
import re
import cv2
from flask import Flask,request,jsonify,abort
from flask_cors import CORS
from io import BytesIO
import base64
import numpy as np
import json
import pandas as pd

app = Flask(__name__)
CORS(app, origins=["*"])

dirnameYolo=os.path.abspath("yoloBest.pt")
model = YOLO(os.path.join(dirnameYolo))
instance = PredictLicensePlate(model)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/predict-image-b64',methods=['POST'])
def universalUploadb64():
    try:
        image_b64 = request.json['img']
        filestr = image_b64.split(',')[1]
        img_bytes = base64.b64decode(filestr)
        img_stream = BytesIO(img_bytes)
        imgArray = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
        res = instance.doPredict(imgArray)
        return res[0]
    except Exception as e:
        error_message = str(e)
        abort(400, description=error_message)
    
@app.route('/predict-image',methods=['POST'])
def universalUpload():
    try:
        file = request.files['file']
        imgArray = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        res = instance.doPredict(imgArray)
        return res[0]
    except Exception as e:
        error_message = str(e)
        abort(400, description=error_message)
    