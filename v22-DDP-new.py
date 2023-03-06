




#v22 


#auc和pf1计算需要注意！！！


#启动方式
#CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python -m torch.distributed.launch --nproc_per_node 6 v22-DDP.py


# ====================================================
# CFG
# ====================================================

class CFG:
	version = 'v22'
	print_freq=100
	num_workers = 12
	model_name = 'efficientnet_b2'
	size = 512
	epochs = 10
	factor = 0.2
	patience = 5
	eps = 1e-6
	lr = 0.0001
	min_lr = 1e-6
	batch_size = 4
	weight_decay = 1e-6
	gradient_accumulation_steps = 6
	max_grad_norm = 1000
	seed = 42
	target_size = 2
	target_col = 'cancer'
	n_fold = 5
	trn_fold = [0,1,2,3,4]
	resize_to = (2048,2048)



# ====================================================
# directory settings
# ====================================================

import os

#OUTPUT_DIR = './output'
OUTPUT_DIR = '../output'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

if not os.path.exists(OUTPUT_DIR+'/log'):
	os.makedirs(OUTPUT_DIR+'/log')

if not os.path.exists(OUTPUT_DIR+'/model'):
	os.makedirs(OUTPUT_DIR+'/model')

if not os.path.exists(OUTPUT_DIR+'/model/'+CFG.version):
	os.makedirs(OUTPUT_DIR+'/model/'+CFG.version)

if not os.path.exists(OUTPUT_DIR+'/submit'):
	os.makedirs(OUTPUT_DIR+'/submit')

if not os.path.exists(OUTPUT_DIR+'/submit/'+CFG.version):
	os.makedirs(OUTPUT_DIR+'/submit/'+CFG.version)

if not os.path.exists(OUTPUT_DIR+'/oof'):
	os.makedirs(OUTPUT_DIR+'/oof')

TRAIN_PATH = '../data/train_images'
TEST_PATH = '../data/test_images'



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
from sklearn.metrics import accuracy_score
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
#from albumentations import (Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose)
#from albumentations.pytorch import ToTensorV2
#from albumentations import ImageOnlyTransform
import timm
import warnings 
warnings.filterwarnings('ignore')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from matplotlib import pyplot as plt
import joblib
from sklearn.model_selection import StratifiedGroupKFold
import torchvision
from sklearn import metrics
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
from contextlib import nullcontext
import argparse

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://')  # nccl是GPU设备上最快、最推荐的后端

# 给主要进程（local_rank=0）设置低输出等级，给其他进程设置高输出等级。
logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)


# ====================================================
# data preprocess
# ====================================================
#train['age'] = train['age'].fillna(58)#平均年龄接近58
#当前版本不使用csv文件的数据，仅使用图片数据



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


@contextmanager
def timer(name):
	t0 = time.time()
	LOGGER.info(f'[{name}] start')
	yield
	LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'/log/train_v'+str(CFG.version)+'.log'):
	from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
	logger = getLogger(__name__)
	logger.setLevel(INFO)
	handler1 = StreamHandler()
	handler1.setFormatter(Formatter("%(message)s"))
	handler2 = FileHandler(filename=log_file)
	handler2.setFormatter(Formatter("%(message)s"))
	logger.addHandler(handler1)
	logger.addHandler(handler2)
	return logger

LOGGER = init_logger()


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
		
		path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
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
				torchvision.transforms.RandomHorizontalFlip(0.5),
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
		self.model = timm.create_model(model_name, pretrained=pretrained)
		
	
	def forward(self, x):
		x = self.model(x)
		return x



# ====================================================
# helper functions
# ====================================================

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



# ====================================================
# train_fn & valid_fn
# ====================================================

# 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
	"""
	Distributed Sampler that subsamples indicies sequentially,
	making it easier to collate all results at the end.
	Even though we only use this sampler for eval and predict (no training),
	which means that the model params won't have to be synced (i.e. will not hang
	for synchronization even if varied number of forward passes), we still add extra
	samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
	to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
	"""
	
	def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
		if num_replicas is None:
			if not torch.distributed.is_available():
				raise RuntimeError("Requires distributed package to be available")
			num_replicas = torch.distributed.get_world_size()
		if rank is None:
			if not torch.distributed.is_available():
				raise RuntimeError("Requires distributed package to be available")
			rank = torch.distributed.get_rank()
		self.dataset = dataset
		self.num_replicas = num_replicas
		self.rank = rank
		self.batch_size = batch_size
		self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
		self.total_size = self.num_samples * self.num_replicas
	
	def __iter__(self):
		indices = list(range(len(self.dataset)))
		# add extra samples to make it evenly divisible
		indices += [indices[-1]] * (self.total_size - len(indices))
		# subsample
		indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
		return iter(indices)
	
	def __len__(self):
		return self.num_samples


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]



def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, scaler):
	#batch_time = AverageMeter()
	#data_time = AverageMeter()
	losses = AverageMeter()
	scores = AverageMeter()
	model.train()
	#start = end = time.time()
	#global_step = 0
	
	for step, (images, labels) in enumerate(train_loader):
		#data_time.update(time.time() - end)
		#images = images.to(device)
		#labels = labels.to(device)
		images = images.to(local_rank)
		labels = labels.to(local_rank)
		batch_size = labels.size(0)
		
		
		#DDP模式梯度累加，间隔一定的step暂时不做梯度同步
		#my_context = model.no_sync if local_rank != -1 and (step+1) % CFG.gradient_accumulation_steps != 0 else nullcontext
		#with my_context():
		with autocast():
			y_preds = model(images)
			#loss = criterion(y_preds, labels) / CFG.gradient_accumulation_steps
			loss = criterion(y_preds, labels)
			#losses.update(loss.item()*CFG.gradient_accumulation_steps, batch_size)
			losses.update(loss.item(), batch_size)
			scaler.scale(loss).backward()
		
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
		
		#if (step + 1) % CFG.gradient_accumulation_steps == 0:
			#optimizer.step()
		scaler.step(optimizer)
		scaler.update()
		optimizer.zero_grad()
		
		#global_step += 1
		
	return losses.avg


def valid_fn(valid_loader, model, criterion, scaler, test_sampler):
	#batch_time = AverageMeter()
	#data_time = AverageMeter()
	losses = AverageMeter()
	scores = AverageMeter()
	model.eval()
	preds = []
	
	#start = end = time.time()
	with torch.no_grad():
		for step, (images, labels) in enumerate(valid_loader):
			#data_time.update(time.time() - end)
			#images = images.to(device)
			#labels = labels.to(device)
			images = images.to(local_rank)
			labels = labels.to(local_rank)
			batch_size = labels.size(0)
			
			with autocast():
				y_preds = model(images)
				loss = criterion(y_preds, labels)
				losses.update(loss.item(), batch_size)
			
			preds.append(y_preds.softmax(1))
			
	#predictions = np.concatenate(preds)
	predictions = distributed_concat(torch.concat(preds, dim=0), len(test_sampler.dataset))
	loss_sum = distributed_concat(torch.tensor([losses.sum]).to(local_rank), len(test_sampler.dataset))
	loss_count = distributed_concat(torch.tensor([losses.count]).to(local_rank), len(test_sampler.dataset))
	
	loss_avg = loss_sum.sum()/loss_count.sum()
	return loss_avg.to('cpu').numpy(), predictions.to('cpu').numpy()



# ====================================================
# train loop
# ====================================================

def train_loop(folds, fold):
	if dist.get_rank() == 0:
		LOGGER.info(f"========== fold: {fold} training ==========")
	
	trn_idx = folds[folds['fold'] != fold].index
	val_idx = folds[folds['fold'] == fold].index
	
	train_folds = folds.loc[trn_idx].reset_index(drop=True)
	valid_folds = folds.loc[val_idx].reset_index(drop=True)
	
	train_processed_path = "../data/train_images_raw_ROI_"+str(CFG.resize_to[0])
	valid_processed_path = "../data/train_images_raw_ROI_"+str(CFG.resize_to[0])
	
	#train_dataset = BreastCancerDataset(train_folds, train_processed_path, transforms=get_transforms(data='train'))
	#valid_dataset = BreastCancerDataset(valid_folds, valid_processed_path, transforms=get_transforms(data='valid'))
	train_dataset = BreastCancerDataset(train_folds, train_processed_path, transforms=get_transforms(aug=True))
	valid_dataset = BreastCancerDataset(valid_folds, valid_processed_path, transforms=get_transforms(aug=False))
	
	
	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	# DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
	#      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
	train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
								num_workers=4, sampler=train_sampler, pin_memory=True, drop_last=True)
	
	test_sampler = SequentialDistributedSampler(valid_dataset, batch_size=CFG.batch_size)
	
	valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
								num_workers=4, sampler=test_sampler, pin_memory=True, drop_last=True)
	#valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
	#							shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
	#train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
	#						  shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
	#valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
	#						  shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
	
	model = EffNetb2(CFG.model_name, pretrained=True)
	## 引入SyncBN，这句代码，会将普通BN替换成SyncBN。实现多卡BN同步
	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	
	#model.to(device)
	
	optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
	
	criterion = nn.CrossEntropyLoss().to(local_rank)
	
	best_score = 0.
	best_loss = np.inf
	scaler = GradScaler()
	
	for epoch in range(CFG.epochs):
		train_loader.sampler.set_epoch(epoch)# 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
		start_time = time.time()
		avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, scaler)
		avg_val_loss, preds = valid_fn(valid_loader, model, criterion, scaler, test_sampler)
		
		valid_labels = valid_folds[CFG.target_col].values
		scheduler.step(avg_val_loss)
		score, _, auc = get_score(valid_labels, preds[:,1])
		elapsed = time.time() - start_time
		if dist.get_rank() == 0:
			LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  auc: {auc:.4f}  time: {elapsed:.0f}s')
			LOGGER.info(f'Epoch {epoch+1} - pf1: {score:.5f}')
		if score > best_score:
			best_score = score
			if dist.get_rank() == 0:
				LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
			#torch.save({'model': model.state_dict(), 'preds': preds[:,1]}, OUTPUT_DIR+'/model/'+CFG.version+'/'+f'{CFG.model_name}_fold{fold}_best.pth')
			
			if dist.get_rank() == 0:
				torch.save(model.module.state_dict(), OUTPUT_DIR+'/model/'+CFG.version+'/'+f'{CFG.model_name}_fold{fold}_best.ckpt')
	check_point = torch.load(OUTPUT_DIR+'/model/'+CFG.version+'/'+f'{CFG.model_name}_fold{fold}_best.pth')
	
	valid_folds['preds'] = check_point['preds']
	
	return valid_folds



# ====================================================
# prepare data
# ====================================================

def prepare_data():
	#所有版本共用同一版数据拆分
	if not os.path.exists("../data/pre_train.csv"):
		
		df = pd.read_csv('../data/train.csv')
		
		skf = StratifiedGroupKFold(n_splits=CFG.n_fold)
		for fold, (_, val_) in enumerate(
			skf.split(X=df, y=df.cancer, groups=df.patient_id)
		):
			df.loc[val_, "fold"] = fold
		
		df.to_csv("../data/pre_train.csv", index=False)
	else:
		df = pd.read_csv('../data/pre_train.csv')
	
	return df



# ====================================================
# main function
# ====================================================

def main():
	
	pre_train = prepare_data()
	
	def get_result(result_df):
		preds = result_df['preds'].values
		labels = result_df[CFG.target_col].values
		pf1, thres, auc = get_score(labels, preds)
		if dist.get_rank() == 0:
			LOGGER.info(f'pf1: {pf1:<.5f}')
			LOGGER.info(f'thres: {thres:<.5f}')
			LOGGER.info(f'auc: {auc:<.5f}')
		return thres
	
	oof_df = pd.DataFrame()
	for fold in range(CFG.n_fold):
		if fold in CFG.trn_fold:
			_oof_df = train_loop(pre_train, fold)
			if dist.get_rank() == 0:
				LOGGER.info(f"========== fold: {fold} result ==========")
			_ = get_result(_oof_df)
			oof_df = pd.concat([oof_df, _oof_df])
			
	if dist.get_rank() == 0:
		LOGGER.info(f"========== CV ==========")
		thres = get_result(oof_df)
		oof_df.to_csv(OUTPUT_DIR+'/oof/'+CFG.version+'_oof.csv', index=False)
	
	df_pred = pd.read_csv(OUTPUT_DIR+'/oof/'+CFG.version+'_oof.csv')
	#df_pred['cancer_pred'] = df_pred.preds > thres
	#print('F1 CV score (multiple thresholds):', sklearn.metrics.f1_score(df_pred.cancer, df_pred.cancer_pred))
	df_pred = df_pred.groupby(['patient_id', 'laterality']).agg(#preds是概率
		cancer_max=('preds', 'max'), cancer_mean=('preds', 'mean'), cancer=('cancer', 'max')
	)
	print('pF1 CV score. Mean aggregation, single threshold, auc:', get_score(df_pred.cancer.values, df_pred.cancer_mean.values))
	print('pF1 CV score. Max aggregation, single threshold, auc:', get_score(df_pred.cancer.values, df_pred.cancer_max.values))


if __name__ == '__main__':
	main()













