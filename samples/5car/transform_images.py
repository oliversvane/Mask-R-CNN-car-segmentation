from PIL import Image, ImageOps
import numpy as np
import os
import random
import math
import os.path
import sys
#pip install Pillow==7.1.2


#tempCode:

directory1="./5car/train/"    #Change to source folder fx dataset/train or dataset train
directory3="./5car/mask/"
sizes=256                        #change image size
threshold=15                     #treshhold margin of class dirrence, should be smaller if pixels are mis classed.
trainP=0.7
FileNames=[]
#tempCodeEND
colors5Doors = [[0,0,0], #black
          [20, 100, 20], #Green
          [250, 250, 10], #Yellow
          [20, 20, 250], #Blue
          [10, 250, 250], #Cyan
          [250, 10, 250], #Pink
          [250, 150, 10], #Orange
          [150, 10, 150], #Purple
          [10, 250, 10]] #Lime
colors = [[0,0,0], #Black
          [0, 100, 0], #Green
          [250, 250, 0], #Yellow
          [0, 0, 250], #Blue 
          [10, 250, 250], #Cyan
          [250, 0, 250], #Pink
          [250, 150, 0], #Orange
          [150, 0, 150], #Purple
          [0, 250, 0]] #Lime

# Synth data generation:
def randomHueZoomFlip(filename):
    img = Image.open(directory1+filename)
    trg = Image.open(directory3+filename)
    RZ=random.randint(-30, 30)
    RF=random.randint(1, 2)
    HC=["red","blue","green"]
    HC_val = random.randint(0, 2)
    H_val = random.uniform(0.05, 0.3)
    img=img.convert('RGB')
    layer = Image.new('RGB', img.size, HC[HC_val]) # "hue" selection is done by choosing a color...
    output = Image.blend(img, layer, (H_val))
    if RF==1:
        return output.rotate(RZ),trg.rotate(RZ)
    else:
        return ImageOps.mirror(output.rotate(RZ)),ImageOps.mirror(trg.rotate(RZ))



def saveIMG(n,source=1):
    if source == 1:
        for x in range(5):
            imgXX,trgX=randomHueZoomFlip(n)
            filenameX="trans_"+n.split('.')[0]+"K"+str(x)+".png"
            trgX.save(directory3+filenameX,quality=100, subsampling=0,format="png")
            imgXX.save(directory1+filenameX,quality=100, subsampling=0,format="png")
            FileNames.append(filename+"K"+str(x))     
    return "done with "+n
#Load data

for filename in os.listdir(directory1):
	if filename.endswith(".png"): 
		if not filename.startswith("sti"):
			if not filename.startswith("trans"):
				print(saveIMG(filename))


