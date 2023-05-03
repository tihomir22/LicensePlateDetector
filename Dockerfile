# Definir la imagen base de Python 3
FROM python:3

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido actual de la carpeta de la aplicación en el contenedor
COPY . .

# Exponer el puerto 5000
EXPOSE 5000

# Ejecutar el servidor con gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "--workers", "4"]