from PIL import Image
import os
from predictFnCNN import PredictLicensePlateCNN
import cv2
import re
import pandas as pd

total_count = 0
success_count = 0
carpeta = "imagenesCochesCarteles"
instance = PredictLicensePlateCNN("yoloBest.pt","model_LicensePlate_5")

archivos = os.listdir(carpeta)
df = pd.DataFrame(columns=['fichero analizado', 'matricula detectada', 'fueDetectada'])
for archivo in archivos[0:50]:
    ruta = os.path.join(carpeta, archivo)
    if os.path.isfile(ruta) and archivo.endswith((".jpg", ".jpeg", ".png")):
        nombre = os.path.splitext(archivo)[0].split("_")[0]
        num_matricula = re.findall(r'\d+', nombre)[0]
        imagen = Image.open(ruta)
        imagen = cv2.imread(ruta)
        res = instance.doPredict(imagen)
        print("Nombre del archivo:", nombre)
        print("Res ",res)

        if num_matricula in res:
            print("Correcto!")
            success_count = success_count + 1
        else:
            print("Incorrecto")
        total_count = total_count + 1
        print("Number "+str(total_count))
        df.loc[len(df)]={'fichero_analizado': archivo, 'matricula_detectada': res, 'fue_detectada': num_matricula in res}
        
print("Total "+str(total_count))
print("Correctos "+str(success_count))
print("Win rate ", str((success_count / total_count) * 100) + "%")
df.to_json('result.json')