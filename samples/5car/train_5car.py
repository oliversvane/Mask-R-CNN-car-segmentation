import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.draw
import skimage.io
import tensorflow

np.set_printoptions(threshold=np.inf)
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CarConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "5car"

    # Train on 3 GPU and 4 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 12 (GPUs * images/GPU).
    GPU_COUNT = 3
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # Background + 8 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape. 
    # DOES NOT WORK YET, WE Made all pictures 256x256 in preprocess
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
    
    #Higher learning rate should result in greater varaince in the begnning but better results
    GRADIENT_CLIP_NORM = 10
    LEARNING_RATE = 0.01
    
    #yolo?
    BATCH_SIZE = 6
    
config = CarConfig()
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class CarDataset(utils.Dataset):

    def load_5car(self, dataset_dir, subset):
        """Load a subset of the 5car dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. 
        self.add_class("5car", 1, "part1")
        self.add_class("5car", 2, "part2")
        self.add_class("5car", 3, "part3")
        self.add_class("5car", 4, "part4")
        self.add_class("5car", 5, "part5")
        self.add_class("5car", 6, "part6")
        self.add_class("5car", 7, "part7")
        self.add_class("5car", 8, "part8")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        for filename in os.listdir(dataset_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = dataset_dir+"/"+filename
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
        
                self.add_image(
                        source="5car",
                        image_id=filename,  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height)
        

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        num_classes = 8 #TODO: add dynamic number of classes (8)
        
        # If not a 5car dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "5car":
            print("image not part of 5car dataset!!!!!!!!!!!!!!!!!!!")
            return super(self.__class__, self).load_mask(image_id)

        # Convert greyscale to binary multi layer image
        # Number of layers equal number of classes
        # [height, width, instance_count]

                    
                    
        info = self.image_info[image_id]
        dataset_dir = "./5car/mask/"
        imageX = skimage.io.imread(dataset_dir+str(info["id"]))            
        class_list = list(set(x for l in imageX for x in l))
        class_list.remove(0)
        mask_image=np.zeros([info["height"], info["width"], len(class_list)], 
                        dtype=np.uint8)
        for i,j in enumerate(imageX):
            for ii,jj in enumerate(j):
                if jj!= 0:
                    pixel=np.zeros(len(class_list))
                    pixel[class_list.index(jj)]=1
                    mask_image[i][ii]=pixel

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask_image.astype(np.bool), np.array(class_list, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "5car":
            return info["path"]
        else:
            print("image not part of 5car dataset!!!!!!!!!!!!!!!!!!!")
            super(self.__class__, self).image_reference(image_id)
# Training dataset
dataset_train = CarDataset()
dataset_train.load_5car("./5car", "train")
dataset_train.prepare()

#TODO:change sources to be dynamic




# Validation dataset
dataset_val = CarDataset()
dataset_val.load_5car("./5car", "val")
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
    
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=490, 
            layers="all")


