# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 22:26:31 2016

@author: Ajwahir
"""

import numpy as np
from PIL import Image
import os
import cv2

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d) 

stride = 40
start_x=0
start_y=50

total = 47
nPtrain = 42 # No. of patients
nPtest = total - nPtrain
nI = 120 #No. of images per patient

destfold = '/ajwahir/uts/patches'
fold = '/mnt/mns/train/train' #data folder

ensure_dir(destfold+'/NervePatch')
ensure_dir(destfold+'/nonNervePatch')


for p in range(1,total+1):
        print('Patient no. '+str(p)+'...')
        for i in range(1,nI+1):
        # read and store image
            imgName = fold+'/'+str(p)+'_'+str(i)+'.tif'
            if os.path.exists(imgName):
            # read and store mask variable
                   maskName = fold+'/'+str(p)+'_'+str(i)+'_'+'mask.tif'
                   nerveName = fold+'/'+str(p)+'_'+str(i)+'.tif'
    
                   nerveImage = cv2.imread(nerveName, 0)
                   maskImage = cv2.imread(maskName, 0) 
                   nerveROI = np.zeros((224,224),np.uint8) 
                    
                    
                    
                   if np.sum(maskImage)>0:
                        className = 'NervePatch'
                        # using findContours func to find the none-zero pieces
                        contours, hierarchy = cv2.findContours(maskImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in contours:
                            if cv2.contourArea(cnt)>5:
                                
                                # Build a ROI to crop the Nerve segment
                                x,y,w,h = cv2.boundingRect(cnt)
                                nerveROI=nerveImage[y+h/2-112:y+h/2+112,x+w/2-112:x+w/2+112]
                                fileName = destfold+'/'+className+'/'+str(p)+'_'+str(i)+'.tif'
                                cv2.imwrite(fileName,nerveROI)
                   else:
                        classname = 'nonNervePatch'
                        for horizontal in range(0,4):
                            for vertical in range(0,5):
                                nonNervePatch = nerveImage[start_y+vertical*stride:start_y+2*vertical*stride,start_x+horizontal*stride:start_x+2*horizontal*stride]
                                fileName = destfold+'/'+className+'/'+str(p)+'_'+str(i)+str(horizontal)+'_'+str(vertical)+'.tif'
                                cv2.imwrite(fileName,nonNervePatch)
                                
                                
                            
                           
            
            
    
    
    
