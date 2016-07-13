# Sort images into class and train and valid
import numpy as np
from PIL import Image
import os

total = 47
nPtrain = 42 # No. of patients
nPtest = total - nPtrain
nI = 120 #No. of images per patient
nClasses = 2




fold = '/mnt/mns/train/train' #data folder
destfoldtrain = '/mnt/mns/data/train'
destfoldval = '/mnt/mns/data/val'

os.mkdir(destfoldtrain+'/Nerve')
os.mkdir(destfoldtrain+'/Non_Nerve')

os.mkdir(destfoldval+'/Nerve')
os.mkdir(destfoldval+'/Non_Nerve')

for p in range(1,total+1):
        print('Patient no. '+str(p)+'...')
	for i in range(1,nI+1):
		# read and store image
		imgName = fold+'/'+str(p)+'_'+str(i)+'.tif'
		if os.path.exists(imgName):
		# read and store mask variable
			maskName = fold+'/'+str(p)+'_'+str(i)+'_'+'mask.tif'
			maskImg = Image.open(maskName)
			mask = np.array(maskImg)
			if np.sum(mask)>0:
				className = 'Nerve'
			else:
				className = 'Non_Nerve'

			if p<=nPtrain:
				newImageName = destfoldtrain+'/'+className+'/'+str(p)+'_'+str(i)+'.tif'
			else:
				newImageName = destfoldval+'/'+className+'/'+str(p)+'_'+str(i)+'.tif'

			Image.open(imgName).save(newImageName)
		#maskImg.save(newMaskName)


	











	# start = 1
	# last = 5508
	# fold = '/mnt/mns/test/test' #data folder
	# destfold = '/mnt/mns/data/val'

	# for i in range(start,last+1):

	# imgName = fold+'/'+str(i)+'.tif'

	# 	if os.path.exists(imgName):
	# 	# read and store mask variable
	# 		maskName = fold+'/'+str(i)+'_'+'mask.tif'
	# 		maskImg = Image.open(maskName)
	# 		mask = np.array(maskImg)
	# 		if np.sum(mask)>0:
	# 			className = 'Nerve'
	# 		else:
	# 			className = 'Non_Nerve'

	# 		newImageName = destfold+'/'+className+'/'+str(i)+'.tif'
	# 		newMaskName = destfold+'/'+className+'/'+str(i)+'_'+'mask.tif'
	# 		Image.open(imgName).save(newImageName)















			
