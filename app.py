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

nombre_fichero = "licence_data.yaml"
with open(nombre_fichero, "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
ruta_actual_abs = os.path.abspath("roboflow")
data["path"] = ruta_actual_abs
data["train"] = ruta_actual_abs+"/train/images"
data["val"]= ruta_actual_abs+"/valid/images"

with open(nombre_fichero, "w") as f:
    yaml.dump(data, f)
    
dirnameYolo=os.path.abspath("01/detect/train/weights/best.pt")
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
        stringJson = instance.doPredict(imgArray).to_json(orient='records')
        df_obj = json.loads(stringJson)
        return jsonify(df_obj)
    except Exception as e:
        error_message = str(e)
        abort(400, description=error_message)
    
@app.route('/predict-image',methods=['POST'])
def universalUpload():
    try:
        file = request.files['file']
        imgArray = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        stringJson = instance.doPredict(imgArray).to_json(orient='records')
        df_obj = json.loads(stringJson)
        return jsonify(df_obj)
    except Exception as e:
        error_message = str(e)
        abort(400, description=error_message)
    