# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
######################################################################
import pytesseract
import numpy as np
import cv2
import platform

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

X_resize=220
Y_resize=70

import os
import re
import imutils
from skimage.transform import radon
import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft
from skimage import img_as_uint
import pandas as pd

try:
    from parabolic import parabolic
    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax
    
class PredictLicensePlate:
    
    def __init__(self,loadedModel):
        self.model = loadedModel
        self.class_list = self.model.model.names
        
    def DetectLicenseWithYolov8 (self,img):
        TabcropLicense=[]
        results = self.model.predict(img)
        result=results[0]
        xyxy= result.boxes.xyxy.cpu().numpy()
        confidence= result.boxes.conf.cpu().numpy()
        
        class_id= result.boxes.cls.cpu().numpy().astype(int)
        class_name = [self.class_list[x] for x in class_id]
        sum_output = list(zip(class_name, confidence,xyxy))
        out_image = img.copy()
        for run_output in sum_output :
            label, con, box = run_output
            if label == "vehicle":continue
            cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            TabcropLicense.append(cropLicense)

        return TabcropLicense

    def doPredict(self,image):
        rotationApplied = 0
        TabImgSelect =self.DetectLicenseWithYolov8(image)

        while len(TabImgSelect) == 0 and rotationApplied < 360:
            image=imutils.rotate(image,angle=rotationApplied)
            TabImgSelect =self.DetectLicenseWithYolov8(image)
            rotationApplied = rotationApplied + 15
        if len(TabImgSelect) == 0: 
            raise ValueError("No license plate detected! Check your image.")
        
        image=TabImgSelect[0]  
            
        x_off=3
        y_off=2
                
        x_resize=220
        y_resize=70
                
        Resize_xfactor=1.78
        Resize_yfactor=1.78

        BilateralOption=0
                
        TabLicensesFounded= self.FindLicenseNumber (image, x_off, y_off, x_resize, y_resize, \
                                    Resize_xfactor, Resize_yfactor, BilateralOption)
        return TabLicensesFounded

    def GetRotationImage(self,image):
        I=image
        I = I - mean(I)  # Demean; make the brightness extend above and below zero
        sinogram = radon(I)
        r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
        rotation = argmax(r)
        row = sinogram[:, rotation]
        N = len(row)
        window = blackman(N)
        spectrum = rfft(row * window)
        frequency = argmax(abs(spectrum))
        return rotation, spectrum, frequency

    def FindLicenseNumber (self,gray, x_offset, y_offset, x_resize, y_resize, \
                        Resize_xfactor, Resize_yfactor, BilateralOption):

        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
        X_resize=x_resize
        Y_resize=y_resize
        
        gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
        
        rotation, spectrum, frquency =self.GetRotationImage(gray)
        rotation=90 - rotation
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
            gray=imutils.rotate(gray,angle=rotation)
        
        X_resize=x_resize
        Y_resize=y_resize
        Resize_xfactor=1.5
        Resize_yfactor=1.5
        
        rotation, spectrum, frquency =self.GetRotationImage(gray)
        rotation=90 - rotation
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
            gray=imutils.rotate(gray,angle=rotation)
        
        TabLicensesFounded=[]
        
        gray1 = img_as_uint(gray > gray.mean())
        text = pytesseract.image_to_string(gray1, lang='eng', config='--psm 7 --oem 3')
        text = ''.join(char for char in text if char.isalnum()) 
        text=self.ProcessText(text)
        if self.ProcessText(text) != "":
            TabLicensesFounded.append(text)
    
        return TabLicensesFounded

    def ProcessText(self,text):
        if text is None: return ""
        if len(text)  > 7:
            return text[-7:]
        else:
            return text   


          


      
                 
        