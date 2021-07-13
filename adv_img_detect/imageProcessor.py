from PIL import Image
import pickle
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.stats
import sys
import statistics

from joblib import Parallel,delayed
import random
import time
import pandas as pd
from imgaug import augmenters as iaa

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return "%.3f" % np.where(sd == 0, 0, m/sd)

def evalcall(im,FT,test=0,phase2=False):
    return eval(FT+'(im,0,phase2)')

def Filters(imgs,FT='FT1',test=1,phase2=False):
    listofsnr = [evalcall(im,FT,test,phase2) for im in imgs]
    return listofsnr

def phase1test(ic,FT='FT1',test=1):
    a= Filters(ic,FT)
    return a

def FT1(im,test=1,phase2=False):#medianblur
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    kernel = np.ones((3,3),np.float32)/25
    dst = cv2.medianBlur(img,25)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3=(float(signaltonoise(dst2, axis=None)))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)

def FT2(im,test=1,phase2=False):#GaussianBlur
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    kernel = np.ones((3,3),np.float32)/25
    dst = cv2.GaussianBlur(img,(3,3),0)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)

def FT3(im,test=1,phase2=False):#AverageBlur
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    kernel = np.ones((3,3),np.float32)/25
    dst = cv2.blur(img,(5,5),0)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT4(im,test=1,phase2=False):#Bilateral blur
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img    
    dst = cv2.bilateralFilter(img,6,75,75)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT5(im,test=1,phase2=False):#AdditivePoissonNoise
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    kernel = np.ones((3,3),np.float32)/25
    aug = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
    im_arr = aug.augment_image(img)    
    dst2 = np.abs(img-np.array(im_arr))
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT6(im,test=1,phase2=False):#AdditivePoissonNoise
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)
    im_arr = aug.augment_image(img)    
    dst2 = np.abs(img-np.array(im_arr))
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT7(im,test=1,phase2=False):#Erode
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.erode(img,kernel,iterations = 1)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT8(im,test=1,phase2=False):#Dialte
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.dilate(img,kernel,iterations = 1)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT9(im,test=1,phase2=False):#opening
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT10(im,test=1,phase2=False):#closing
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT11(im,test=1,phase2=False):#Morphology_gradient
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT12(im,test=1,phase2=False):#TopHat
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img=img
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT13(im,test=1,phase2=False):#Blackhat
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        ing=img
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    snr2 = float(signaltonoise(dst, axis=None))
    index = img==dst
    dst2 = np.abs(img-dst)
    snr3 = float(signaltonoise(dst2, axis=None))
    histg = cv2.calcHist([dst2],[0],None,[256],[0,256]) 
    histg=histg[1:127]
    histg=np.average(histg)
    return str(snr3)+","+str(histg)
  
def FT14(im,test=1,phase2=False):#laplacian
    imgm=Image.open(im)
    img = np.array(imgm)
    if test!= 0:
        img = img[:,:,:3]
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    snr2 = float(signaltonoise(laplacian, axis=None))
    return str(snr2)+",0"

def getFilterValues(input_image_path):
    filter_sub = ['a','b','c','d','e','f','g','h']
    output_dict = {}
    phase1test_out = []
    image=[]
    image.append(input_image_path)
    print(f'image {image}')
    for filterNo in range(14):
        filterName='FT'+str(filterNo+1) 
        phase1test_out = phase1test(image,filterName,0)
        for p in range(len(phase1test_out)):
            if type(phase1test_out[0]) is float:
                print("float")
                output_dict[filterName + " a"] = phase1test_out[0]
            else:
                phase1test_out_split = phase1test_out[p].split(",")
                for s in range(len(phase1test_out_split)):
                    output_dict[filterName + " " + filter_sub[s]] = phase1test_out_split[s]
    
    output_df = pd.DataFrame.from_dict(output_dict,orient='index').T
    return output_df     
