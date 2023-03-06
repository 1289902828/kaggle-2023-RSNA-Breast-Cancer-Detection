


#原图数据处理+ROI提取



import numpy as np
import cv2
import os
from joblib import Parallel, delayed
from pathlib import Path
import dicomsdl
import multiprocessing as mp

#RESIZE_TO = (2048, 2048)#根据原生图像大小决定

#directories = list(Path('../input/rsna-breast-cancer-detection/train_images').iterdir())
directories = list(Path('/hy-tmp/data/train_images').iterdir())


# https://www.kaggle.com/code/tanlikesmath/brain-tumor-radiogenomic-classification-eda/notebook
def dicom_file_to_ary(path):
	dcm_file = dicomsdl.open(str(path))
	data = dcm_file.pixelData()
	
	data = (data - data.min()) / (data.max() - data.min())
	
	if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
		data = 1 - data
	
	#data = cv2.resize(data, RESIZE_TO)
	data = (data * 255).astype(np.uint8)
	return data


def process_directory(directory_path):
	parent_directory = str(directory_path).split('/')[-1]
	#!mkdir -p /kaggle/temp/test_images_processed_cv2_dicomsdl_{RESIZE_TO[0]}/{parent_directory}
	!mkdir -p ../data/train_images_processed_cv2_dicomsdl_raw/{parent_directory}
	for image_path in directory_path.iterdir():
		processed_ary = dicom_file_to_ary(image_path)
		
		cv2.imwrite(
			#f'/kaggle/temp/test_images_processed_cv2_dicomsdl_{RESIZE_TO[0]}/{parent_directory}/{image_path.stem}.png',
			f'../data/train_images_processed_cv2_dicomsdl_raw/{parent_directory}/{image_path.stem}.png',
			processed_ary
		)

with mp.Pool(12) as p:
	p.map(process_directory, directories)

#处理用时1小时20分





#ROI提取（基于已经转换格式的数据做）

import numpy as np
import cv2
import os
from pathlib import Path
import dicomsdl
import multiprocessing as mp


d_roi = list(Path('/hy-tmp/data/train_images_processed_cv2_dicomsdl_raw').iterdir())
#d_roi = list(Path('/kaggle/temp/test_images_processed_cv2_dicomsdl_2048').iterdir())

#RESIZE_TO = (1024,1024)


#先滤波再检测轮廓
def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)#高斯滤波
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #检测物体轮廓
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


#def get_roi(path):
#	
#	img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)#读成灰度图像
#	img = cv2.imread(str(path))
#	(x, y, w, h) = crop_coords(img_gray)
#	
#	img = img[y:y+h, x:x+w]
#	
#	#img = cv2.resize(img, RESIZE_TO)
#	img = (img * 255).astype(np.uint8)
#	return img


def get_roi(path):
	
	img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)#读成灰度图像
	#img = cv2.imread(str(path))
	(x, y, w, h) = crop_coords(img_gray)
	
	img = img_gray[y:y+h, x:x+w]
	#img = cv2.resize(img, RESIZE_TO)
	#img = (img * 255).astype(np.uint8)
	img = img.astype(np.uint8)
	return img


def process_directory(directory_path):
	parent_directory = str(directory_path).split('/')[-1]
	
	!mkdir -p ../data/train_images_raw_ROI/{parent_directory}
	for image_path in directory_path.iterdir():
		roi_ary = get_roi(image_path)
		
		cv2.imwrite(
			#f'/kaggle/temp/test_images_ROI_{RESIZE_TO[0]}_{RESIZE_TO[1]}/{parent_directory}/{image_path.stem}.png',
			f'../data/train_images_raw_ROI/{parent_directory}/{image_path.stem}.png',
			roi_ary
		)


with mp.Pool(12) as p:
	p.map(process_directory, d_roi)

#不到1小时


#统计ROI图像大小，随机抽2000张

import numpy as np
import cv2
import dicomsdl
import random
import pandas as pd

df = pd.read_csv('../data/pre_train.csv')
df = df.sample(frac=1).reset_index(drop=True)

shape_0 = []
shape_1 = []

for i in range(0,2000):
	#image_path = '/hy-tmp/data/train_images/'+str(df.iloc[i].patient_id)+'/'+str(df.iloc[i].image_id)+'.dcm'
	image_path = '/hy-tmp/data/train_images_raw_ROI/'+str(df.iloc[i].patient_id)+'/'+str(df.iloc[i].image_id)+'.png'
	data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	
	shape_0.append(data.shape[0])
	shape_1.append(data.shape[1])



#正样本shape分布
import numpy as np
import cv2
import dicomsdl
import random
import pandas as pd

df = pd.read_csv('../data/pre_train.csv')
df_p = df[df['cancer']==0].reset_index(drop=True)

shape_0 = []
shape_1 = []

for i in range(0,df_p.shape[0]):
	#image_path = '/hy-tmp/data/train_images/'+str(df.iloc[i].patient_id)+'/'+str(df.iloc[i].image_id)+'.dcm'
	image_path = '/hy-tmp/data/train_images_raw_ROI/'+str(df.iloc[i].patient_id)+'/'+str(df.iloc[i].image_id)+'.png'
	data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	
	shape_0.append(data.shape[0])
	shape_1.append(data.shape[1])

plt.hist(shape_0,50)
plt.show()

plt.hist(shape_1,50)
plt.show()




#复制正样本

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

train = pd.read_csv('/hy-tmp/data/train.csv')
df = train[train['cancer']==1].reset_index(drop=True)

!mkdir /hy-tmp/data/positive_png
for i in range(df.shape[0]):
	image_path = '/hy-tmp/data/train_images_raw_ROI/'+str(df.iloc[i].patient_id)+'/'+str(df.iloc[i].image_id)+'.png'
	!cp {image_path} /hy-tmp/data/positive_png






