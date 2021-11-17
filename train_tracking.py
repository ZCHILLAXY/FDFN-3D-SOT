import argparse
import os
import random
import time
import logging
import pdb

from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from model.loss import rpn_cross_entropy_balance, reg_smoothL1, box_iou3d, focal_loss, criterion_smoothl1, \
	depth_smoothL1
from test_tracking import test
from utils.anchors import cal_rpn_target, cal_anchors

from loader.Dataset import SiameseTrain, SiameseTest
from model.model import SiamPillar
from utils.metrics import AverageMeter
from config import cfg


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--learning_rate', type=float, default=0.0004, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default = 0,  help='number of input point features')
parser.add_argument('--data_dir', type=str, default = '/home/lilium/zhuange/kitti/training',  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Cyclist',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='results_cyclist',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')

opt = parser.parse_args()

#torch.cuda.set_device(opt.main_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = opt.save_root_dir

try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
def tracking_collate(batch):

	t_vox_feature = []
	t_vox_number = []
	t_vox_coordinate = []
	s_vox_feature = []
	s_vox_number = []
	s_vox_coordinate = []
	rgb_t_vox_feature = []
	rgb_t_vox_number = []
	rgb_t_vox_coordinate = []
	rgb_s_vox_feature = []
	rgb_s_vox_number = []
	rgb_s_vox_coordinate = []
	gt_RGB = []
	sample_RGB = []
	template_box = []
	sample_box = []
	gt_box_lst = []

	for i, data in enumerate(batch):
		t_vox_feature.append(data[0])
		t_vox_number.append(data[1])
		t_vox_coordinate.append(np.pad(data[2], ((0, 0), (1, 0)), mode = 'constant', constant_values = i))
		s_vox_feature.append(data[3])
		s_vox_number.append(data[4])
		s_vox_coordinate.append(np.pad(data[5], ((0, 0), (1, 0)), mode = 'constant', constant_values = i))
		rgb_t_vox_feature.append(data[6])
		rgb_t_vox_number.append(data[7])
		rgb_t_vox_coordinate.append(np.pad(data[8], ((0, 0), (1, 0)), mode='constant', constant_values=i))
		rgb_s_vox_feature.append(data[9])
		rgb_s_vox_number.append(data[10])
		rgb_s_vox_coordinate.append(np.pad(data[11], ((0, 0), (1, 0)), mode = 'constant', constant_values = i))
		gt_RGB.append(data[12])
		sample_RGB.append(data[13])
		template_box.append(data[14])
		sample_box.append(data[15])
		gt_box_lst.append(data[16])

	return torch.from_numpy(np.concatenate(t_vox_feature, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(t_vox_number, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(t_vox_coordinate, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(s_vox_feature, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(s_vox_number, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(s_vox_coordinate, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(rgb_t_vox_feature, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(rgb_t_vox_number, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(rgb_t_vox_coordinate, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(rgb_s_vox_feature, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(rgb_s_vox_number, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(rgb_s_vox_coordinate, axis=0)).float(), \
		   torch.from_numpy(np.array(gt_RGB)).float(), \
		   torch.from_numpy(np.array(sample_RGB)).float(), \
		   torch.from_numpy(np.concatenate(template_box, axis=0)).float(), \
		   torch.from_numpy(np.concatenate(sample_box, axis=0)).float(), \
		   np.array(gt_box_lst)

train_data = SiameseTrain(
            input_size=512,
            path= opt.data_dir,
            split='Train',
            category_name=opt.category_name,
            offset_BB=0,
            scale_BB=1.25)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
	collate_fn=tracking_collate,
    pin_memory=True)

test_data = SiameseTrain(
    input_size=512,
    path=opt.data_dir,
    split='Valid',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers / 2),
	collate_fn=tracking_collate,
    pin_memory=True)

dataset_Test = SiameseTest(
	input_size=512,
	path=opt.data_dir,
	split='Test',
	category_name=opt.category_name,
	offset_BB=0,
	scale_BB=1.25)

test_loader = torch.utils.data.DataLoader(
	dataset_Test,
	collate_fn=lambda x: x,
	batch_size=1,
	shuffle=False,
	num_workers=0,
	pin_memory=True)

Success_run = AverageMeter()
Precision_run = AverageMeter()

										  
print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
model = SiamPillar()
if opt.ngpu > 1:
	model = torch.nn.DataParallel(model, range(opt.ngpu))
if opt.model != '':
	model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)), strict=False)
	  
model.cuda()
print(model)


optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas = (0.9, 0.999), eps=1e-08)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.4)

# 3. Training and testing
for epoch in range(opt.nepoch):
	scheduler.step(epoch)
	print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
#	# 3.1 switch to train mode
	torch.cuda.synchronize()
	model.train()
	train_mse = 0.0
	timer = time.time()

	batch_correct = 0.0
	batch_cla_loss = 0.0
	batch_reg_loss = 0.0
	batch_cla_pos_loss = 0.0
	batch_cla_neg_loss = 0.0
	batch_label_loss = 0.0
	batch_center_loss = 0.0
	batch_theta_loss = 0.0
	batch_regularization_loss = 0.0
	batch_num = 0.0
	batch_iou = 0.0
	batch_true_correct = 0.0
	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets
		t_vox_feature, t_vox_number, t_vox_coordinate, \
		s_vox_feature, s_vox_number, s_vox_coordinate, \
		rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate, \
		rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate, \
		gt_RGB, sample_RGB, template_box, sample_box, gt_box_lst = data

		t_vox_feature = Variable(t_vox_feature, requires_grad=False).cuda()
		t_vox_number = Variable(t_vox_number, requires_grad=False).cuda()
		t_vox_coordinate = Variable(t_vox_coordinate, requires_grad=False).cuda()
		s_vox_feature = Variable(s_vox_feature, requires_grad=False).cuda()
		s_vox_number = Variable(s_vox_number, requires_grad=False).cuda()
		s_vox_coordinate = Variable(s_vox_coordinate, requires_grad=False).cuda()
		rgb_t_vox_feature = Variable(rgb_t_vox_feature, requires_grad=False).cuda()
		rgb_t_vox_number = Variable(rgb_t_vox_number, requires_grad=False).cuda()
		rgb_t_vox_coordinate = Variable(rgb_t_vox_coordinate, requires_grad=False).cuda()
		rgb_s_vox_feature = Variable(rgb_s_vox_feature, requires_grad=False).cuda()
		rgb_s_vox_number = Variable(rgb_s_vox_number, requires_grad=False).cuda()
		rgb_s_vox_coordinate = Variable(rgb_s_vox_coordinate, requires_grad=False).cuda()
		gt_RGB = Variable(gt_RGB, requires_grad=False).cuda()
		sample_RGB = Variable(sample_RGB, requires_grad=False).cuda()
		template_box = Variable(template_box, requires_grad=False).cuda()
		sample_box = Variable(sample_box, requires_grad=False).cuda()

		anchors = cal_anchors()  # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)
		#z_pos_equal_one, z_targets, z_depths = cal_rpn_target(gt_box_lst, [cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT], anchors, dim='z')
		#y_pos_equal_one, y_targets = cal_rpn_target(sample_box, [cfg.FEATURE_DEPTH, cfg.FEATURE_HEIGHT], anchors, dim='y')
		x_pos_equal_one, x_targets, x_depths = cal_rpn_target(gt_box_lst, [cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH], anchors, dim='x')


		#pos_equal_one = cal_scoremap(sample_box, [cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT])
		# z_pos_equal_one = torch.from_numpy(z_pos_equal_one).float()
		# z_targets = torch.from_numpy(z_targets).float()
		# z_depths = torch.from_numpy(z_depths).float()
		# y_pos_equal_one = torch.from_numpy(y_pos_equal_one).float()
		# y_targets = torch.from_numpy(y_targets).float()
		x_pos_equal_one = torch.from_numpy(x_pos_equal_one).float()
		x_targets = torch.from_numpy(x_targets).float()
		x_depths = torch.from_numpy(x_depths).float()
		# z_pos_equal_one = Variable(z_pos_equal_one, requires_grad=False).cuda()
		# z_targets = Variable(z_targets, requires_grad=False).cuda()
		# z_depths = Variable(z_depths, requires_grad=False).cuda()
		# y_pos_equal_one = Variable(y_pos_equal_one, requires_grad=False).cuda()
		# y_targets = Variable(y_targets, requires_grad=False).cuda()
		x_pos_equal_one = Variable(x_pos_equal_one, requires_grad=False).cuda()
		x_targets = Variable(x_targets, requires_grad=False).cuda()
		x_depths = Variable(x_depths, requires_grad=False).cuda()

		gt_center = Variable(torch.from_numpy(gt_box_lst[:, 0:3]).float(), requires_grad=False).cuda()
		gt_theta = Variable(torch.from_numpy(gt_box_lst[:, 6]).float().unsqueeze(1), requires_grad=False).cuda()

		# 3.1.2 compute output

		pred_conf, pred_reg, pred_depth, final_coord, final_angle = model(len(gt_box_lst), t_vox_feature, t_vox_number, t_vox_coordinate,
																							 s_vox_feature, s_vox_number, s_vox_coordinate,
																							rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate, rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate,
																							 gt_RGB, sample_RGB, template_box, sample_box)

		# z_cls_loss, z_pcls_loss, z_ncls_loss = focal_loss(pred_conf, z_pos_equal_one)
		# y_cls_loss, y_pcls_loss, y_ncls_loss = focal_loss(y_pred_conf, y_pos_equal_one)
		cls_loss, pcls_loss, ncls_loss = focal_loss(pred_conf, x_pos_equal_one)

		#cls_loss, pcls_loss, ncls_loss = rpn_cross_entropy_balance(pred_conf, pos_equal_one)
		# z_reg_loss = reg_smoothL1(pred_reg, z_targets, z_pos_equal_one)
		# z_depth_loss = depth_smoothL1(pred_depth, z_depths, z_pos_equal_one)
		# y_reg_loss = rpn_smoothL1(y_pred_reg, y_targets, y_pos_equal_one)
		reg_loss = reg_smoothL1(pred_reg, x_targets, x_pos_equal_one)
		depth_loss = depth_smoothL1(pred_depth, x_depths, x_pos_equal_one)

		center_loss = criterion_smoothl1(final_coord, gt_center)
		theta_loss = criterion_smoothl1(final_angle, gt_theta)
		# box_loss = criterion_smoothl1(pj_roi_boxes, rgb_roi_boxes)
		#loss_label = criterion_cla(pred_seed, label_cla)
		#loss_box = criterion_box(pred_offset, label_reg)
		#loss_box = (loss_box.mean(2) * label_cla).sum()/(label_cla.sum()+1e-06)

		regularization_loss = 0
		# for offset_name, offset_param in model.Offset_Head.named_parameters():
		# 	if 'mask' in offset_name:
		# 		continue
		# 	regularization_loss += torch.sum(torch.abs(offset_param))
		# for angle_name, angle_param in model.Angle_Head.named_parameters():
		# 	if 'mask' in angle_name:
		# 		continue
		# 	regularization_loss += torch.sum(torch.abs(angle_param))
		cls_loss = cls_loss
		reg_loss = reg_loss + 0.2 * depth_loss
		pcls_loss = pcls_loss
		ncls_loss = ncls_loss

		loss = cls_loss + 5 * reg_loss + 2 * (center_loss + theta_loss) #+ 0.001 * regularization_loss

		# 3.1.3 compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		torch.cuda.synchronize()
		
		# 3.1.4 update training error
		# estimation_cla_cpu = seed_pediction.sigmoid().detach().cpu().numpy()
		# label_cla_cpu = label_cla.detach().cpu().numpy()
		# correct = float(np.sum((estimation_cla_cpu[0:len(label_point_set),:] > 0.4) == label_cla_cpu[0:len(label_point_set),:])) / 169.0
		# true_correct = float(np.sum((np.float32(estimation_cla_cpu[0:len(label_point_set),:] > 0.4) + label_cla_cpu[0:len(label_point_set),:]) == 2)/(np.sum(label_cla_cpu[0:len(label_point_set),:])))
					
		train_mse = train_mse + loss.data*len(sample_box)
		# batch_correct += correct
		batch_cla_loss += cls_loss.data
		batch_reg_loss += reg_loss.data
		batch_cla_pos_loss += pcls_loss
		batch_cla_neg_loss += ncls_loss
		batch_center_loss += center_loss.data
		batch_theta_loss += theta_loss.data
		batch_regularization_loss = regularization_loss
		# batch_num += len(label_point_set)
		# batch_true_correct += true_correct
		if (i+1)%20 == 0:
			print('\n ---- batch: %03d ----' % (i+1))
			print('cla_loss: %f, reg_loss: %f, cla_pos_loss: %f, cls_neg_loss: %f, center_loss: %f, theta_loss: %f, l1_loss: %f'%
				  (batch_cla_loss/20, batch_reg_loss/20, batch_cla_pos_loss/20, batch_cla_neg_loss/20, batch_center_loss/20, batch_theta_loss/20, batch_regularization_loss/20))
			# print('accuracy: %f' % (batch_correct / float(batch_num)))
			# print('true accuracy: %f' % (batch_true_correct / 20))
			batch_label_loss = 0.0
			batch_cla_loss = 0.0
			batch_reg_loss = 0.0
			batch_cla_pos_loss = 0.0
			batch_cla_neg_loss = 0.0
			batch_center_loss = 0.0
			batch_theta_loss = 0.0
			batch_num = 0.0
			batch_true_correct = 0.0
           
	# time taken
	train_mse = train_mse/len(train_data)
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	torch.save(model.state_dict(), '%s/model_%d.pth' % (save_dir, epoch))
	#torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))
	
	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	model.eval()
	test_cla_loss = 0.0
	test_reg_loss = 0.0
	test_cla_pos_loss = 0.0
	test_cla_neg_loss = 0.0
	test_label_loss = 0.0
	test_center_loss = 0.0
	test_theta_loss = 0.0
	test_regularization_loss = 0.0
	test_correct = 0.0
	test_true_correct = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(test_dataloader, 0)):
		torch.cuda.synchronize()
		# 3.2.1 load inputs and targets
		t_vox_feature, t_vox_number, t_vox_coordinate, \
		s_vox_feature, s_vox_number, s_vox_coordinate, \
		rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate, \
		rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate, \
		gt_RGB, sample_RGB, template_box, sample_box, gt_box_lst = data

		t_vox_feature = Variable(t_vox_feature, requires_grad=False).cuda()
		t_vox_number = Variable(t_vox_number, requires_grad=False).cuda()
		t_vox_coordinate = Variable(t_vox_coordinate, requires_grad=False).cuda()
		s_vox_feature = Variable(s_vox_feature, requires_grad=False).cuda()
		s_vox_number = Variable(s_vox_number, requires_grad=False).cuda()
		s_vox_coordinate = Variable(s_vox_coordinate, requires_grad=False).cuda()
		rgb_t_vox_feature = Variable(rgb_t_vox_feature, requires_grad=False).cuda()
		rgb_t_vox_number = Variable(rgb_t_vox_number, requires_grad=False).cuda()
		rgb_t_vox_coordinate = Variable(rgb_t_vox_coordinate, requires_grad=False).cuda()
		rgb_s_vox_feature = Variable(rgb_s_vox_feature, requires_grad=False).cuda()
		rgb_s_vox_number = Variable(rgb_s_vox_number, requires_grad=False).cuda()
		rgb_s_vox_coordinate = Variable(rgb_s_vox_coordinate, requires_grad=False).cuda()
		gt_RGB = Variable(gt_RGB, requires_grad=False).cuda()
		sample_RGB = Variable(sample_RGB, requires_grad=False).cuda()
		template_box = Variable(template_box, requires_grad=False).cuda()
		sample_box = Variable(sample_box, requires_grad=False).cuda()

		anchors = cal_anchors()  # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)
		# z_pos_equal_one, z_targets, z_depths = cal_rpn_target(gt_box_lst, [cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT],
		# 													  anchors, dim='z')
		# y_pos_equal_one, y_targets = cal_rpn_target(sample_box, [cfg.FEATURE_DEPTH, cfg.FEATURE_HEIGHT], anchors, dim='y')
		x_pos_equal_one, x_targets, x_depths = cal_rpn_target(gt_box_lst, [cfg.FEATURE_WIDTH, cfg.FEATURE_WIDTH], anchors, dim='x')

		# pos_equal_one = cal_scoremap(sample_box, [cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT])
		# z_pos_equal_one = torch.from_numpy(z_pos_equal_one).float()
		# z_targets = torch.from_numpy(z_targets).float()
		# z_depths = torch.from_numpy(z_depths).float()
		# y_pos_equal_one = torch.from_numpy(y_pos_equal_one).float()
		# y_targets = torch.from_numpy(y_targets).float()
		x_pos_equal_one = torch.from_numpy(x_pos_equal_one).float()
		x_targets = torch.from_numpy(x_targets).float()
		x_depths = torch.from_numpy(x_depths).float()
		# z_pos_equal_one = Variable(z_pos_equal_one, requires_grad=False).cuda()
		# z_targets = Variable(z_targets, requires_grad=False).cuda()
		# z_depths = Variable(z_depths, requires_grad=False).cuda()
		# y_pos_equal_one = Variable(y_pos_equal_one, requires_grad=False).cuda()
		# y_targets = Variable(y_targets, requires_grad=False).cuda()
		x_pos_equal_one = Variable(x_pos_equal_one, requires_grad=False).cuda()
		x_targets = Variable(x_targets, requires_grad=False).cuda()
		x_depths = Variable(x_depths, requires_grad=False).cuda()

		gt_center = Variable(torch.from_numpy(gt_box_lst[:, 0:3]).float(), requires_grad=False).cuda()
		gt_theta = Variable(torch.from_numpy(gt_box_lst[:, 6]).float().unsqueeze(1), requires_grad=False).cuda()

		# 3.1.2 compute output

		pred_conf, pred_reg, pred_depth, final_coord, final_angle = model(len(gt_box_lst), t_vox_feature, t_vox_number,
																		  t_vox_coordinate,
																		  s_vox_feature, s_vox_number, s_vox_coordinate,
																		  rgb_t_vox_feature, rgb_t_vox_number,
																		  rgb_t_vox_coordinate, rgb_s_vox_feature,
																		  rgb_s_vox_number, rgb_s_vox_coordinate,
																		  gt_RGB, sample_RGB, template_box, sample_box)

		# z_cls_loss, z_pcls_loss, z_ncls_loss = focal_loss(pred_conf, z_pos_equal_one)
		# y_cls_loss, y_pcls_loss, y_ncls_loss = focal_loss(y_pred_conf, y_pos_equal_one)
		cls_loss, pcls_loss, ncls_loss = focal_loss(pred_conf, x_pos_equal_one)

		# cls_loss, pcls_loss, ncls_loss = rpn_cross_entropy_balance(pred_conf, pos_equal_one)
		# z_reg_loss = reg_smoothL1(pred_reg, z_targets, z_pos_equal_one)
		# z_depth_loss = depth_smoothL1(pred_depth, z_depths, z_pos_equal_one)
		# y_reg_loss = rpn_smoothL1(y_pred_reg, y_targets, y_pos_equal_one)
		reg_loss = reg_smoothL1(pred_reg, x_targets, x_pos_equal_one)
		depth_loss = depth_smoothL1(pred_depth, x_depths, x_pos_equal_one)

		center_loss = criterion_smoothl1(final_coord, gt_center)
		theta_loss = criterion_smoothl1(final_angle, gt_theta)
		# box_loss = criterion_smoothl1(pj_roi_boxes, rgb_roi_boxes)
		# loss_label = criterion_cla(pred_seed, label_cla)
		# loss_box = criterion_box(pred_offset, label_reg)
		# loss_box = (loss_box.mean(2) * label_cla).sum()/(label_cla.sum()+1e-06)

		#loss_label = criterion_cla(pred_seed, label_cla)
		#loss_box = criterion_box(pred_offset, label_reg)
		#loss_box = (loss_box.mean(2) * label_cla).sum() / (label_cla.sum() + 1e-06)

		regularization_loss = 0
		# for offset_name, offset_param in model.Offset_Head.named_parameters():
		# 	if 'mask' in offset_name:
		# 		continue
		# 	regularization_loss += torch.sum(torch.abs(offset_param))
		# for angle_name, angle_param in model.Angle_Head.named_parameters():
		# 	if 'mask' in angle_name:
		# 		continue
		# 	regularization_loss += torch.sum(torch.abs(angle_param))
		cls_loss = cls_loss
		reg_loss = reg_loss + 0.2 * depth_loss
		pcls_loss = pcls_loss
		ncls_loss = ncls_loss

		loss = cls_loss + 5 * reg_loss + 2 * (center_loss + theta_loss) #+ 0.01 * regularization_loss

		torch.cuda.synchronize()
		test_cla_loss = test_cla_loss + cls_loss.data*len(sample_box)
		test_reg_loss = test_reg_loss + reg_loss.data*len(sample_box)
		test_cla_pos_loss = test_cla_pos_loss + pcls_loss.data*len(sample_box)
		test_cla_neg_loss = test_cla_neg_loss + ncls_loss.data*len(sample_box)
		test_center_loss = test_center_loss + center_loss.data*len(sample_box)
		test_theta_loss = test_theta_loss + theta_loss.data * len(sample_box)
		test_regularization_loss = test_regularization_loss + regularization_loss * len(sample_box)
		# estimation_cla_cpu = seed_pediction.sigmoid().detach().cpu().numpy()
		# label_cla_cpu = label_cla.detach().cpu().numpy()
		# correct = float(np.sum((estimation_cla_cpu[0:len(label_point_set),:] > 0.4) == label_cla_cpu[0:len(label_point_set),:])) / 169.0
		# true_correct = float(np.sum((np.float32(estimation_cla_cpu[0:len(label_point_set),:] > 0.4) + label_cla_cpu[0:len(label_point_set),:]) == 2)/(np.sum(label_cla_cpu[0:len(label_point_set),:])))
		# test_correct += correct
		# test_true_correct += true_correct*len(label_point_set)

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	# print mse
	test_cla_loss = test_cla_loss / len(test_data)
	test_reg_loss = test_reg_loss / len(test_data)
	test_cla_pos_loss = test_cla_pos_loss / len(test_data)
	test_cla_neg_loss = test_cla_neg_loss / len(test_data)
	test_label_loss = test_label_loss / len(test_data)
	test_center_loss = test_center_loss / len(test_data)
	test_theta_loss = test_theta_loss / len(test_data)
	test_regularization_loss = test_regularization_loss / len(test_data)
	print('cla_loss: %f, reg_loss: %f, center_loss: %f, angle_loss: %f, l1_loss: %f, #test_data = %d' %(test_cla_loss, test_reg_loss, test_center_loss, test_theta_loss, test_regularization_loss, len(test_data)))
	# test_correct = test_correct / len(test_data)
	# print('mean-correct of 1 sample: %f, #test_data = %d' %(test_correct, len(test_data)))
	# test_true_correct = test_true_correct / len(test_data)
	# print('true correct of 1 sample: %f' %(test_true_correct))
	# log
	logging.info('Epoch#%d: train error=%e, test error=%e, %e, %e, %e, %e lr = %f' %(epoch, train_mse, test_cla_loss, test_reg_loss, test_center_loss, test_theta_loss, test_regularization_loss, scheduler.get_lr()[0]))

	# Succ, Prec = test(
	# 	test_loader,
	# 	model,
	# 	epoch=epoch + 1,
	# 	shape_aggregation='firstandprevious',
	# 	reference_BB='previous_result',
	# 	IoU_Space=3)
	# Success_run.update(Succ)
	# Precision_run.update(Prec)
	# logging.info("mean Succ/Prec {}/{}".format(Success_run.avg, Precision_run.avg))
	#
	# Success_run.reset()
	# Precision_run.reset()

