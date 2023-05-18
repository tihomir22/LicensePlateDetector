from PIL import Image
import os
from predictFn import PredictLicensePlate
import cv2
import re
import pandas as pd
import os
from ultralytics import YOLO
from io import BytesIO
import numpy as np

total_count = 0
success_count = 0
carpeta = "imagenesCochesCarteles"
instance = PredictLicensePlate(YOLO(os.path.join("yoloBest.pt")))

archivos = os.listdir(carpeta)
df = pd.DataFrame(columns=['fichero_analizado', 'matricula_detectada', 'fue_detectada'],data=[])
for archivo in archivos:
    ruta = os.path.join(carpeta, archivo)
    if os.path.isfile(ruta) and archivo.endswith((".jpg", ".jpeg", ".png")):
        nombre = os.path.splitext(archivo)[0].split("_")[0]
        num_matricula = re.findall(r'\d+', nombre)[0]
        imagen = cv2.imread(ruta)
        img_stream = BytesIO(imagen)
        imgArray = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
        try:
            res = instance.doPredict(imagen)['Matricula detectada'][0]
            print("Nombre del archivo:", nombre)
            print("Res ",res)

            if num_matricula in res or num_matricula is res:
                print("Correcto!")
                success_count = success_count + 1
            else:
                print("Incorrecto")
            total_count = total_count + 1
            print("Number "+str(total_count))
            # df.loc[len(df)]={'fichero_analizado': archivo, 'matricula_detectada': res, 'fue_detectada': num_matricula in res}
        except Exception as e:
            print("ERROR " + str(e))
            df.loc[len(df)]={'fichero_analizado': archivo, 'matricula_detectada': 'ERROR', 'fue_detectada': False}
        
print("Total "+str(total_count))
print("Correctos "+str(success_count))
print("Win rate ", str((success_count / total_count) * 100) + "%")
df = df[df['fue_detectada'] == False]
df.to_json('result.json')