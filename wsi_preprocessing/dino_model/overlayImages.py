import cv2
import os
path = './DINO_VISUALIZATION6/'
files = os.listdir(path)
files.sort()
img = cv2.imread(path + files[-1])
files.remove(files[-1])

for f in files:
    attn = cv2.imread(path + f)
    added_image = cv2.addWeighted(img,0.7,attn,0.6,0)
    cv2.imwrite(path + f + '_comb.png', added_image)
