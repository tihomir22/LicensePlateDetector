from PIL import Image
import os
from predictFnCNN import PredictLicensePlateCNN
import cv2
import re

total_count = 0
success_count = 0
carpeta = "imagenesCochesCarteles"
instance = PredictLicensePlateCNN("yoloBest.pt","model_LicensePlate_4")

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta)

# Recorrer los archivos
for archivo in archivos:
    # Ruta completa del archivo
    ruta = os.path.join(carpeta, archivo)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Configura el nivel de registro de TensorFlow a "ERROR" para evitar impresiones
    # Verificar si es un archivo de imagen
    if os.path.isfile(ruta) and archivo.endswith((".jpg", ".jpeg", ".png")):
        # Obtener el nombre del archivo sin la extensión
        nombre = os.path.splitext(archivo)[0].split("_")[0]
        num_matricula = re.findall(r'\d+', nombre)[0]
        
        # Cargar la imagen usando PIL
        imagen = Image.open(ruta)
        
        # Realizar operaciones con la imagen aquí
        # Por ejemplo, mostrar el nombre del archivo
        
        imagen = cv2.imread(ruta)
        #imgArray = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        res = instance.doPredict(imagen)
        print("Nombre del archivo:", nombre)
        print("Res ",res)

        if num_matricula in res:
            print("Correcto!")
            success_count = success_count + 1
        else:
            print("Incorrecto")
        total_count = total_count + 1
        # Cerrar la imagen
        cv2.destroyAllWindows()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
print("Total "+str(total_count))
print("Correctos "+str(success_count))
print("Win rate ", str((success_count / total_count) * 100) + "%")