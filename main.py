
# from flask import Flask, render_template, url_for, request
# import pandas as pd
# import numpy as np
# import pandas as pd
# import numpy as np
# import IPython


# app = Flask(__name__)
# @app.after_request
# def add_header(response):
#     response.cache_control.max_age = 0
#     return response



# @app.route('/')
# def get_image():
#     image = request.form['image']
#     elapsed_time = predict_segments(image)
#     return render_template('main.html',elapsed_time)




# #|--------------------------------------------------------------
# #|                                                            |
# #|     COCO Config Files                                                       |
# #|                                                            |
# #--------------------------------------------------------------
# #!%tensorflow_version 1.x
# import os
# import sys
# import random
# import math
# import re
# import time
# import numpy as np
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import json
# import os
# import shutil
# import zipfile
# import tensorflow as tf
# from mrcnn.config import Config
# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log
# from PIL import Image, ImageDraw
# import warnings
# warnings.filterwarnings('ignore')
# import matplotlib.pyplot as plt
# import mrcnn.model as model_mask_lib
# import imgaug
# from pycocotools import mask
# from coco import CocoDataset
# from coco import CocoConfig


# class Brain_Config(Config):
#   def __init__(self):
#     super().__init__()
#   NAME = "teeth_train"
#   GPU_COUNT = 1
#   IMAGES_PER_GPU = 1
#   NUM_CLASSES = 1+1
#   IMAGE_MAX_DIM = 512
#   IMAGE_MIN_DIM = 512
#   STEPS_PER_EPOCH = 500
#   DETECTION_MIN_CONFIDENCE = 0.9 
  
# def load_datasets():
#     dataset_train = CocoDataset()
#     dataset_train.load_coco("/content/MRI_Tumor", "train", year=2021, return_coco=True, auto_download=False)
#     dataset_train.prepare()
#     dataset_test = CocoDataset()
#     dataset_test.load_coco("/content/MRI_Tumor", "test", year=2021, return_coco=True, auto_download=False)
#     dataset_test.prepare()
#     return dataset_train,dataset_test

# def get_ax(rows=1, cols=1, size=8):
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax

# def predict_segments(image):
#     start = time.time()
#     dataset_train , dataset_test = load_datasets()
#     inf_config = Brain_Config()
#     model = model_mask_lib.MaskRCNN(mode='inference',config=inf_config,model_dir="/contents/backup")
#     model.load_weights('Brain_weights.h5',by_name=True)
#     results = model.detect([image], verbose=1)
#     r = results[0]
#     visualize.save_image(image,r['masks'],r['class_ids'],r['scores'],save_dir='static/result.jpg')
#     return(time.time() - start)

# if name__=='__main__':
#     app.run(debug=True,port=8080)

import tensorflow as tf
print(tf.__version__)