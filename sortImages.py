# Sort images into class and train and valid

import numpy as np
from PIL import Image
import os

nP = 47 # No. of patients
nI = 120 #No. of images per patient
nClasses = 2
fold = 'G:/athi/acads/kaggleUS/train' #data folder

for p in range(1,nP):
	for i in range(1,nI):
		# read and store image
		imgName = fold+'/'+str(p)+'_'+str(i)+'.tif'
		# read and store mask variable
		maskName = fold+'/'+str(p)+'_'+str(i)+'_'+'mask.tif'
		mask = Image.open(maskName)
		mask = np.array(mask)
		if np.sum(mask)>0:
			
