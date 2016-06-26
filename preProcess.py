# Read images, preprocess, write to numpy file
# As of now, only histogram equalization is implemented

import numpy as np
from PIL import Image

# histogram equalization
def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape)



nP = 47 # No. of patients
nI = 120 #No. of images per patient
fold = 'G:/athi/acads/kaggleUS/train' #data folder
foldSave = 'G:/athi/acads/kaggleUS/GT' #folder to save numpy file

imgs = np.zeros((nP*nI,420,580))
targets = np.zeros(nP*nI)
k = 1

for p in range(1,nP):
	for i in range(1,nI):
		# read and store image
		imgName = fold+'/'+str(p)+'_'+str(i)+'.tif'
		img = Image.open(imgName)
		img = np.array(img)
		# img = histeq(img)  #histogram equalization?
		imgs[k,:,:] = img

		# Pre-process

		# read and store mask variable
		maskName = fold+'/'+str(p)+'_'+str(i)+'_'+'mask.tif'
		mask = Image.open(maskName)
		mask = np.array(mask)
		if np.sum(mask)>0:
			targets[k] = 1

		k = k+1

#save
data = (imgs,targets)
np.save(foldSave+'/raw_data.npy',data)

# check
check = 1
if check:
	img_obj = Image.fromarray(imgs[0,:,:].reshape((420,580)))
	img_obj.save('imgCheck.jpg')






