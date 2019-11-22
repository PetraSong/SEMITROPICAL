import cv2 as cv
import numpy as np
import glob
import statistics
from PIL import Image

blist = []
glist = []
rlist = []

for img in glob.glob("AE*.png"):
	imgop = cv.imread(img)
	blist.append(cv.mean(imgop)[0])
	glist.append(cv.mean(imgop)[1])
	rlist.append(cv.mean(imgop)[2])

print("b channel mean and standard deviation: ")
print(statistics.mean(blist))
print(np.std(blist))
bmean = statistics.mean(blist)
bstd = np.std(blist)

print("g channel mean and standard deviation: ")
print(statistics.mean(glist))
print(np.std(glist))
gmean = statistics.mean(glist)
gstd = np.std(glist)

print("r channel mean and standard deviation: ")
print(statistics.mean(rlist))
print(np.std(rlist))
rmean = statistics.mean(rlist)
rstd = np.std(rlist)

#standardization operation: subtract mean and divide by standard deviation each channel

print("standardization commencing... ")

for img in glob.glob("AE*.png"):
	imgop = cv.imread(img)
	print("shapes: ")
	print(imgop.shape)
	bchan, gchan, rchan = cv.split(imgop)
	#print(bchan, gchan, rchan)
	bchanNorm = (bchan-bmean)/bstd
	gchanNorm = (gchan-gmean)/gstd
	rchanNorm = (rchan-rmean)/rstd
	#scale back up to 0-255 scale
	bchanStd = 255*(bchanNorm-np.min(bchanNorm))/np.ptp(bchanNorm).astype(int)
	gchanStd = 255*(gchanNorm-np.min(gchanNorm))/np.ptp(gchanNorm).astype(int)
	rchanStd = 255*(rchanNorm-np.min(rchanNorm))/np.ptp(rchanNorm).astype(int)
	#print(bchanNorm[0], gchanNorm[0], rchanNorm[0])	
	stdImg = cv.merge((bchanStd,gchanStd,rchanStd))
	
	print(stdImg)
	cv.imwrite("std_"+img,stdImg)
