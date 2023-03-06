


#数据处理最终方案



#dicom->png
#ROI
#resize

#上传的时候要改版本，resize


#线下
############################################################################################


import numpy as np
import cv2
import os
from pathlib import Path
import dicomsdl
import multiprocessing as mp


d_roi = list(Path('/hy-tmp/data/train_images_processed_cv2_dicomsdl_raw').iterdir())
#d_roi = list(Path('/kaggle/temp/test_images_processed_cv2_dicomsdl_2048').iterdir())

class CFG:
	RESIZE_TO = (2048,2048)


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


def get_roi(path):
	
	img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)#读成灰度图像
	#img = cv2.imread(str(path))
	(x, y, w, h) = crop_coords(img_gray)
	
	img = img_gray[y:y+h, x:x+w]
	
	img = cv2.resize(img, CFG.RESIZE_TO)
	#img = cv2.resize(img, (1536,2560))
	
	img = img.astype(np.uint8)
	return img


def process_directory(directory_path):
	parent_directory = str(directory_path).split('/')[-1]
	
	!mkdir -p ../data/train_images_raw_ROI_{CFG.RESIZE_TO[1]}/{parent_directory}
	for image_path in directory_path.iterdir():
		roi_ary = get_roi(image_path)
		
		cv2.imwrite(
			f'../data/train_images_raw_ROI_{CFG.RESIZE_TO[1]}/{parent_directory}/{image_path.stem}.png',
			roi_ary
		)


with mp.Pool(12) as p:
	p.map(process_directory, d_roi)




#正样本上采样见对应文件


#线上批式处理
##########################################################################################




#参数
COMP_FOLDER = '/kaggle/input/rsna-breast-cancer-detection/'
DATA_FOLDER = COMP_FOLDER + 'test_images/'

#N_CORES = mp.cpu_count()
#MIXED_PRECISION = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RAM_CHECK = True
DEBUG = True

test_df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')
#test_df['cancer'] = 0 #dummy value



PUBLIC_RUN = False

if PUBLIC_RUN is False:
    RAM_CHECK = False
    DEBUG = False

if RAM_CHECK is True:
    test_df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')
    patient_filter = list(sorted((set(test_df.patient_id.unique()))))[:100]
    test_df = test_df[test_df.patient_id.isin(patient_filter)]
    DATA_FOLDER = DATA_FOLDER.replace('test','train')

if DEBUG is True:
    test_df = test_df.head(100)

test_df["fns"] = test_df['patient_id'].astype(str) + '/' + test_df['image_id'].astype(str) + '.dcm'

#------------------------------------------------------------------------------------
#数据处理

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


def get_roi(img_gray):
	
	#img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)#读成灰度图像，这里读一次即可
	#img = cv2.imread(str(path))
	(x, y, w, h) = crop_coords(img_gray)
	
	img = img_gray[y:y+h, x:x+w]
	
	img = cv2.resize(img, (2048,2048))#需要改
	#img = (img * 255).astype(np.uint8)
	img = img.astype(np.uint8)
	return img


def convert_dicom_to_jpg(file, save_folder=""):
	patient = file.split('/')[-2]
	image = file.split('/')[-1][:-4]
	dcmfile = pydicom.dcmread(file)
	
	if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
		with open(file, 'rb') as fp:
			raw = DicomBytesIO(fp.read())
			ds = pydicom.dcmread(raw)
		offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
		hackedbitstream = bytearray()
		hackedbitstream.extend(ds.PixelData[offset:])
		with open(save_folder + f"{patient}_{image}.jpg", "wb") as binary_file:
			binary_file.write(hackedbitstream)
			
	if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.70':
		with open(file, 'rb') as fp:
			raw = DicomBytesIO(fp.read())
			ds = pydicom.dcmread(raw)
		offset = ds.PixelData.find(b"\xff\xd8\xff\xe0")  #<---- the jpeg lossless header info we're looking for
		hackedbitstream = bytearray()
		hackedbitstream.extend(ds.PixelData[offset:])
		with open(save_folder + f"{patient}_{image}.jpg", "wb") as binary_file:
			binary_file.write(hackedbitstream)


@pipeline_def
def jpg_decode_pipeline(jpgfiles):
    jpegs, _ = fn.readers.file(files=jpgfiles)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16)
    return images


def process_dicom(img, dicom):
	try:
		invert = getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1"
	except:
		invert = False
	
	img = (img - img.min()) / (img.max() - img.min())
	
	if invert:
		img = 1 - img
	
	return img


def process(f, save_folder=""):
	patient = f.split('/')[-2]
	dicom_id = f.split('/')[-1][:-4]
	
	dicom = dicomsdl.open(f)
	img = dicom.pixelData()
	img = torch.from_numpy(img)
	img = process_dicom(img, dicom)
	
	#img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), (SAVE_SIZE, SAVE_SIZE), mode="bilinear")[0, 0]
	
	img = (img * 255).clip(0,255).to(torch.uint8).cpu().numpy()
	out_file_name = SAVE_FOLDER + f"{patient}_{dicom_id}.png"
	cv2.imwrite(out_file_name, img)
	return out_file_name




# ====================================================
# CFG
# ====================================================

class CFG:
	version = 'v19'
	#print_freq=100
	#num_workers = 12
	model_name = 'efficientnet_b2'
	#size = 1024
	#epochs = 10
	#factor = 0.2
	#patience = 5
	#eps = 1e-6
	#lr = 1e-4
	#min_lr = 1e-6
	batch_size = 6
	#weight_decay = 1e-6
	#gradient_accumulation_steps = 1
	#max_grad_norm = 1000
	seed = 42
	target_size = 2
	target_col = 'cancer'
	#n_fold = 5
	#trn_fold = [0,1,2,3,4]
	resize_to = (1024,1024)



# ====================================================
# libraries
# ====================================================

import sys
import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import preprocessing
#from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from functools import partial
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import timm
import warnings 
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from matplotlib import pyplot as plt
import joblib
from sklearn.model_selection import StratifiedGroupKFold
import torchvision
from sklearn import metrics
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler



# ====================================================
# utils
# ====================================================

def pfbeta(labels, predictions, beta=1.):
	y_true_count = 0
	ctp = 0
	cfp = 0
	
	for idx in range(len(labels)):
		prediction = min(max(predictions[idx], 0), 1)
		if (labels[idx]):
			y_true_count += 1
			ctp += prediction
		else:
			cfp += prediction
	
	beta_squared = beta * beta
	c_precision = ctp / (ctp + cfp)
	c_recall = ctp / max(y_true_count, 1)  # avoid / 0
	if (c_precision > 0 and c_recall > 0):
		result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
		return result
	else:
		return 0


def get_score(labels, predictions):
	auc = metrics.roc_auc_score(labels, predictions)
	thres = np.linspace(0, 1, 1001)
	f1s = [pfbeta(labels, predictions > thr) for thr in thres]
	idx = np.argmax(f1s)
	return f1s[idx], thres[idx], auc


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)



# ====================================================
# dataset
# ====================================================

class BreastCancerDataset(Dataset):
	
	def __init__(self, df, path, transforms=None):
		super().__init__()
		self.df = df
		self.path = path
		self.transforms = transforms
	
	def __getitem__(self, i):
		
		path = f'{self.path}/{self.df.iloc[i].patient_id}_{self.df.iloc[i].image_id}.png'
		try:
			img = Image.open(path).convert('RGB')
		except Exception as ex:
			print(path, ex)
			return None
		
		if self.transforms is not None:
			img = self.transforms(img)
		
		if CFG.target_col in self.df.columns:
			cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
			#cat_aux_targets = torch.as_tensor(self.df.iloc[i][CATEGORY_AUX_TARGETS])
			#return img, cancer_target, cat_aux_targets
			return img, cancer_target
		
		return img
	
	def __len__(self):
		return len(self.df)



# ====================================================
# Data Augmentation
# ====================================================
#后续版本需要优化

def get_transforms(aug=False):
	
	def transforms(img):
		#img = img.convert('RGB')#.resize((512, 512))
		if aug:
			tfm = [
				torchvision.transforms.RandomHorizontalFlip(0.5),
				torchvision.transforms.RandomRotation(degrees=(-5, 5)),
				#torchvision.transforms.RandomResizedCrop((512, 512), scale=(0.8, 1), ratio=(1, 1))
			]
		else:
			tfm = [
				#torchvision.transforms.RandomHorizontalFlip(0.5),
				#torchvision.transforms.Resize((1024, 512))
			]
		img = torchvision.transforms.Compose(tfm + [
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=0.2179, std=0.0529),
			
		])(img)
		return img
	
	return lambda img: transforms(img)



# ====================================================
# model initialization
# ====================================================

class EffNetb2(nn.Module):
	def __init__(self, model_name, pretrained=False):
		super().__init__()
		self.model = timm.create_model(model_name, pretrained=False)
		
	
	def forward(self, x):
		x = self.model(x)
		return x



def load_model(name, dir='.', model=None):
	state = torch.load(os.path.join(dir, f'{name}'), map_location=device)
	if model is None:
		model = EffNetb2(CFG.model_name)
	model.load_state_dict(state['model'])
	# print(data['threshold'], data['model_type'])
	return model


models = []

for fname in tqdm(sorted(os.listdir('/kaggle/input/model-'+CFG.version))):
	
	model = load_model(fname, '/kaggle/input/model-'+CFG.version)
	model = model.to(device)
	models.append(model)
	print(f'fname:{fname}')


def models_predict(models, ds, max_batches=1e9):
	dl_test = torch.utils.data.DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=os.cpu_count())
	for m in models:
		m.eval()
	
	with torch.no_grad():
		predictions = []
		for idx, X in enumerate(tqdm(dl_test, mininterval=30)):
			pred = torch.zeros(len(X), len(models))
			for idx, m in enumerate(models):
				with autocast():
					
					y_preds = m.forward(X.to(device))
					
				pred[:, idx] = y_preds.softmax(1)[:,1].to('cpu')
				
			predictions.append(pred.mean(dim=-1))
			
			if idx >= max_batches:
				break
		return torch.concat(predictions).numpy()



#以上都不需要改
#-------------------------------------------------------------------------

#We will process the dicoms in chunks so the disk space does not become an issue.
#GPU处理部分

#SAVE_SIZE = 2048
SAVE_FOLDER = "/tmp/output/"

N_CHUNKS = 40 if len(test_df["fns"]) > 100 else 1
#100张一批
CHUNKS = [(len(test_df["fns"]) / N_CHUNKS * k, len(test_df["fns"]) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
CHUNKS = np.array(CHUNKS).astype(int)
JPG_FOLDER = "/tmp/jpg/"

test_pre = []

#-----------------------------------------------------------------------------

for ttt, chunk in enumerate(CHUNKS):
	print(f'chunk {ttt} of {len(CHUNKS)} chunks')
	os.makedirs(JPG_FOLDER, exist_ok=True)
	os.makedirs(SAVE_FOLDER, exist_ok=True)
	_ = Parallel(n_jobs=2)(
		delayed(convert_dicom_to_jpg)(f'{DATA_FOLDER}/{img}', save_folder=JPG_FOLDER)
		for img in test_df["fns"].tolist()[chunk[0]: chunk[1]]
	)
	gc.collect()
	torch.cuda.empty_cache()
	
	jpgfiles = glob.glob(JPG_FOLDER + "*.jpg")
	
	pipe = jpg_decode_pipeline(jpgfiles, batch_size=1, num_threads=2, device_id=0)
	pipe.build()
	
	for i, f in enumerate(tqdm(jpgfiles)):
		
		patient, dicom_id = f.split('/')[-1][:-4].split('_')
		dicom = pydicom.dcmread(DATA_FOLDER + f"/{patient}/{dicom_id}.dcm")
		try:
			out = pipe.run()
			# Dali -> Torch
			img = out[0][0]
			img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
			feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
			img = img_torch.float()
			
			#apply dicom preprocessing
			img = process_dicom(img, dicom)
			del dicom
			#resize the torch image
			#img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), (SAVE_SIZE, SAVE_SIZE), mode="bilinear")[0, 0]
			img = torch.squeeze(img,2)
			img = (img * 255).clip(0,255).to(torch.uint8).cpu().numpy()
			
			#ROI
			img = get_roi(img)
			
			
			out_file_name = SAVE_FOLDER + f"{patient}_{dicom_id}.png"
			cv2.imwrite(out_file_name, img)
			
		
		except Exception as e:
			if i == len(jpgfiles)-1:
				continue
			else:
				#print(i, e)
				pipe = jpg_decode_pipeline(jpgfiles[i+1:], batch_size=1, num_threads=2, device_id=0)
				pipe.build()
				continue
	
	gc.collect()
	torch.cuda.empty_cache()
	
	#CPU部分(还需要改，统一一下预测结果保存格式)
	fns = glob.glob(f'{SAVE_FOLDER}/*.png')
	gpu_processed_files = [fn.split('/')[-1].replace('_','/').replace('png','dcm') for fn in fns]
	to_process = [f for f in test_df["fns"][chunk[0]: chunk[1]].values if f not in gpu_processed_files]
	
	cpu_processed_filenames = Parallel(n_jobs=2)(
		delayed(process)(f'{DATA_FOLDER}/{img}', save_folder=SAVE_FOLDER)
		for img in tqdm(to_process)
	)
	
	gc.collect()
	torch.cuda.empty_cache()
	
	#推理
	
	
	#df_test = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')
	df_test = test_df[chunk[0]:chunk[1]].copy().reset_index(drop=True)
	
	test_processed_path = "/tmp/output"
	
	ds_test = BreastCancerDataset(df_test, test_processed_path, transforms=get_transforms(aug=False))
	
	models_pred = models_predict(models, ds_test)
	
	#df_test['cancer'] = models_pred
	test_pre.extend(list(models_pred))
	
	
	shutil.rmtree(JPG_FOLDER)
	shutil.rmtree(SAVE_FOLDER)
#print(f'DALI Raw image load complete')


gc.collect()
torch.cuda.empty_cache()

#submission

#选一个
THRES = 0.033 # mean
#THRES = 0.058 #max
test_df['cancer'] = test_pre

df_sub = test_df[['prediction_id', 'cancer']].groupby('prediction_id').mean().reset_index()

THRES = np.quantile(df_sub.cancer.values,0.97935)

df_sub['cancer'] = (df_sub.cancer > THRES).astype(int)

df_sub.to_csv('submission.csv', index=False)






