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

def loadImagesFromDirname (dirname):
     images = []
     print("Reading imagenes from ",dirname)
     Cont=0
     for root, dirnames, filenames in os.walk(dirname):
         for filename in filenames:
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 filepath = os.path.join(root, filename)
                 image = cv2.imread(filepath)
                 img = image
                 r, g, b = cv2.split(img)
                 r_avg = cv2.mean(r)[0]
                 g_avg = cv2.mean(g)[0]
                 b_avg = cv2.mean(b)[0]
                 # Find the gain occupied by each channel
                 k = (r_avg + g_avg + b_avg)/3
                 kr = k/r_avg
                 kg = k/g_avg
                 kb = k/b_avg
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                 balance_img = cv2.merge([b, g, r])
                 image=balance_img
                 images.append(image)
                 Cont+=1
     return images,filenames
imagesComplete,filenames=loadImagesFromDirname(os.path.join(os.path.abspath("roboflow/test_clandestino")))
print("Number of imagenes : " + str(len(imagesComplete)))


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/predict-image-b64',methods=['POST'])
def universalUploadb64():
    image_b64 = request.json['img']
    filestr = image_b64.split(',')[1]
    img_bytes = base64.b64decode(filestr)
    img_stream = BytesIO(img_bytes)
    imgArray = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    stringJson = instance.doPredict(imgArray).to_json(orient='records')
    df_obj = json.loads(stringJson)
    return jsonify(df_obj)
    
@app.route('/predict-image',methods=['POST'])
def universalUpload():
    file = request.files['file']
    imgArray = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    stringJson = instance.doPredict(imgArray).to_json(orient='records')
    df_obj = json.loads(stringJson)
    return jsonify(df_obj)
    