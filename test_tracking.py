import math
import time
import os
import logging
import argparse
import random

import cv2
import numpy as np
import matplotlib
import json

import matplotlib.pyplot as plt
from mayavi import mlab
from pyquaternion import Quaternion
from tqdm import tqdm

import torch

import utils.common_utils as utils
import copy
from datetime import datetime

from model.model import SiamPillar
from utils.draw_utils import draw_lidar, draw_gt_boxes3d, draw_rgb_projections
from utils.metrics import AverageMeter, Success, Precision
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.data_classes import PointCloud, Box
from loader.Dataset import SiameseTest

import torch.nn.functional as F
from torch.autograd import Variable

from utils.spconv_preprocess import spconv_process_pointcloud, spconv_process_project

from config import cfg


def test(loader,model,epoch=-1,shape_aggregation="",reference_BB="",max_iter=-1,IoU_Space=3):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()
    Success_batch = Success()
    Precision_batch = Precision()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    dataset = loader.dataset
    batch_num = 0
    c_50=0
    c_100=0
    c_200=0
    cl_50, pl_50 = [], []
    cl_100, pl_100 = [], []
    cl_200, pl_200 = [], []

    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno)) as t:
        for batch in loader:          
            batch_num = batch_num+1
            # measure data loading time
            data_time.update((time.time() - end))
            for PCs, RGBs, BBs, Calibs, list_of_anno in batch:  # tracklet
                results_BBs = []
                results_success = []
                results_precision = []
                with open('bat_car/bat_car_2.json') as f:
                    d = json.load(f)
                for i, _ in enumerate(PCs):
                    this_anno = list_of_anno[i]
                    this_BB = BBs[i]
                    this_PC = PCs[i]
                    this_RGB = RGBs[i]
                    this_Calib = Calibs[i]
                    this_PJ_Matrix = utils.getPJMatrix(this_Calib)
                    gt_boxs = []
                    result_boxs = []

                    # INITIAL FRAME
                    if i == 0:
                        box = BBs[i]
                        results_BBs.append(box)
                        box_z = 0
                        box_wlh = box.wlh
                        #first_PC = utils.getModel([this_PC], [this_BB], offset=dataset.offset_BB, scale=dataset.scale_BB)

                    else:
                        previous_BB = BBs[i - 1]
                        # DEFINE REFERENCE BB
                        if ("previous_result".upper() in reference_BB.upper()):
                            ref_BB = results_BBs[-1]
                        elif ("previous_gt".upper() in reference_BB.upper()):
                            ref_BB = previous_BB
                            # ref_BB = utils.getOffsetBB(this_BB,np.array([-1,1,1]))
                        elif ("current_gt".upper() in reference_BB.upper()):
                            ref_BB = this_BB
                        candidate_PC, sample_box, trans, rot_mat = utils.cropAndCenterPC_new(
                                        this_PC,
                                        ref_BB,this_BB,
                                        offset=dataset.offset_BB,
                                        scale=dataset.scale_BB)

                        this_box_center = sample_box[0:3]
                        this_box_size = sample_box[3:6]
                        this_box_rot = Quaternion(axis=[0, 0, 1], radians=sample_box[6])
                        new_gt_box = Box(center=this_box_center,
                                  size=this_box_size,
                                  orientation=this_box_rot)




                        candidate_PC = utils.regularizePC_scene(candidate_PC, dataset.input_size,istrain=False)
                        target_center = np.expand_dims(sample_box[0:3], axis=0)
                        candidate_PC = np.concatenate([candidate_PC, target_center], axis=0)

                        candidate_RGB = utils.cropRGB_test(this_RGB, ref_BB, this_PJ_Matrix, search=True, offset=dataset.offset_BB,
                                                            scale=dataset.scale_BB)
                        sample_forward_PC = utils.cropForwardPC(this_PC, this_RGB, candidate_RGB, ref_BB,
                                                                this_PJ_Matrix, search=True, offset=dataset.offset_BB,
                                                                scale=dataset.scale_BB)
                        sample_forward_PC = utils.regularizePC_scene(sample_forward_PC, dataset.input_size,istrain=False)
                        candidate_RGB = cv2.resize(candidate_RGB, (cfg.SCENE_INPUT_WIDTH, cfg.SCENE_INPUT_WIDTH),
                                                cv2.INTER_NEAREST)

                        scene_voxel_dict = spconv_process_pointcloud(candidate_PC)
                        rgb_scene_voxel_dict = spconv_process_project(sample_forward_PC)

                        # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                        if ("firstandprevious".upper() in shape_aggregation.upper()):
                            # tracklet_length = len(PCs)
                            # first_ratio = - (i ** 4  / (tracklet_length * 2) ** 4) + 1
                            # new_pts_idx = np.random.randint(low=0, high=PCs[0].points.shape[1], size=int(first_ratio * PCs[0].points.shape[1]), dtype=np.int64)
                            # # previous_ratio = 1 - first_ratio
                            # PCs[0] = PointCloud(PCs[0].points[:, new_pts_idx])
                            # PCs[i - 1] = PointCloud(PCs[i - 1].points[:, 0:int(PCs[i - 1].points.shape[1] * previous_ratio)])
                            model_PC = utils.getModel([PCs[0], PCs[i-1]], [results_BBs[0],results_BBs[i-1]],offset=dataset.offset_BB,scale=dataset.scale_BB)
                            if i==1:
                                logging.info(model_PC.points.shape)
                        elif ("first".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0]], [results_BBs[0]],offset=dataset.offset_BB,scale=dataset.scale_BB)
                        elif ("previous".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[i-1]], [results_BBs[i-1]],offset=dataset.offset_BB,scale=dataset.scale_BB)
                        elif ("all".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel(PCs[:i],results_BBs,offset=dataset.offset_BB,scale=dataset.scale_BB)
                        else:
                            model_PC = utils.getModel(PCs[:i],results_BBs,offset=dataset.offset_BB,scale=dataset.scale_BB)
                        model_PC = utils.regularizePC_template(model_PC, dataset.input_size, istrain=False)
                        object_center = np.zeros((1, 3))
                        model_PC = np.concatenate([model_PC, object_center], axis=0)

                        prev_RGB = RGBs[i - 1]
                        gt_Calib_pre = Calibs[i - 1]
                        gt_PJ_Matrix = utils.getPJMatrix(gt_Calib_pre)

                        gt_RGB = utils.cropRGB_test(prev_RGB, results_BBs[i - 1], gt_PJ_Matrix)
                        gt_forward_PC = utils.cropForwardPC(PCs[i-1], prev_RGB, gt_RGB, this_BB, gt_PJ_Matrix)
                        gt_forward_PC = utils.regularizePC_template(gt_forward_PC, dataset.input_size,istrain=False)

                        gt_RGB = cv2.resize(gt_RGB, (cfg.TEMPLATE_INPUT_WIDTH, cfg.TEMPLATE_INPUT_WIDTH),
                                            cv2.INTER_NEAREST)

                        # u, v, z = gt_forward_PC.T
                        # plt.imshow(gt_RGB)
                        # # plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
                        # plt.show()

                        template_voxel_dict = spconv_process_pointcloud(model_PC, template=True)
                        rgb_template_voxel_dict = spconv_process_project(gt_forward_PC, template=True)

                        t_vox_feature = torch.from_numpy(template_voxel_dict['feature_buffer']).float()
                        t_vox_number = torch.from_numpy(template_voxel_dict['number_buffer']).float()
                        t_vox_coordinate = torch.from_numpy(np.pad(template_voxel_dict['coordinate_buffer'], ((0, 0), (1, 0)), mode = 'constant', constant_values = 0)).float()
                        s_vox_feature = torch.from_numpy(scene_voxel_dict['feature_buffer']).float()
                        s_vox_number = torch.from_numpy(scene_voxel_dict['number_buffer']).float()
                        s_vox_coordinate = torch.from_numpy(
                            np.pad(scene_voxel_dict['coordinate_buffer'], ((0, 0), (1, 0)), mode='constant',
                                   constant_values=0)).float()
                        rgb_t_vox_feature = torch.from_numpy(rgb_template_voxel_dict['feature_buffer']).float()
                        rgb_t_vox_number = torch.from_numpy(rgb_template_voxel_dict['number_buffer']).float()
                        rgb_t_vox_coordinate = torch.from_numpy(
                            np.pad(rgb_template_voxel_dict['coordinate_buffer'], ((0, 0), (1, 0)), mode='constant',
                                   constant_values=0)).float()
                        rgb_s_vox_feature = torch.from_numpy(rgb_scene_voxel_dict['feature_buffer']).float()
                        rgb_s_vox_number = torch.from_numpy(rgb_scene_voxel_dict['number_buffer']).float()
                        rgb_s_vox_coordinate = torch.from_numpy(
                            np.pad(rgb_scene_voxel_dict['coordinate_buffer'], ((0, 0), (1, 0)), mode='constant',
                                   constant_values=0)).float()

                        template_box = np.expand_dims(
                            np.array([0, 0, 0, sample_box[3], sample_box[4], sample_box[5], 0]), 0).repeat(
                            t_vox_feature.shape[0], axis=0)
                        sample_box = np.expand_dims(np.array(sample_box), 0).repeat(s_vox_feature.shape[0], axis=0)

                        candidate_RGB = torch.from_numpy(np.array(candidate_RGB)).float().unsqueeze(0)
                        gt_RGB = torch.from_numpy(np.array(gt_RGB)).float().unsqueeze(0)
                        template_box = torch.from_numpy(template_box).float()
                        sample_box = torch.from_numpy(sample_box).float()

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
                        candidate_RGB = Variable(candidate_RGB, requires_grad=False).cuda()
                        gt_RGB = Variable(gt_RGB, requires_grad=False).cuda()
                        template_box = Variable(template_box, requires_grad=False).cuda()
                        sample_box = Variable(sample_box, requires_grad=False).cuda()


                        model.track_init(t_vox_feature, t_vox_number, t_vox_coordinate,
                                         rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate,
                                         gt_RGB, template_box, box_wlh)

                        estimation_box = model.track(s_vox_feature, s_vox_number, s_vox_coordinate,
                                                     rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate,
                                                     candidate_RGB, sample_box)
                        #estimation_box = estimation_box.cpu().detach().numpy()
                        # box_idx = estimation_boxs_cpu[:,4].argmax()
                        # estimation_box_cpu = estimation_boxs_cpu[box_idx,0:4]
                        #
                        b_center = estimation_box[0:3]
                        b_size = box_wlh
                        b_rot = Quaternion(axis=[0, 0, 1], radians=estimation_box[6])
                        box = Box(center=b_center,
                                  size=b_size,
                                  orientation=b_rot)
                        final_box = copy.deepcopy(box)
                        final_box.rotate(Quaternion(matrix=(np.transpose(rot_mat))))
                        final_box.translate(-trans)
                        results_BBs.append(final_box)


                        # box_dir = os.path.join(f'{this_anno["type"]}_tracking_results', str(this_anno['scene']))
                        # if not os.path.exists(box_dir):
                        #     os.makedirs(box_dir)
                        # predict_file = os.path.join(box_dir, str(this_anno['track_id']) + f'_{this_anno["frame"]}' + '_predict.txt')
                        # truth_file = os.path.join(box_dir, str(this_anno['track_id']) + f'_{this_anno["frame"]}' + '_truth.txt')
                        # with open(predict_file, encoding='utf-8', mode='a') as f1, open(truth_file, encoding='utf-8',
                        #                                                                 mode='a') as f2:
                        #     line_predict = ','.join(
                        #         [str(final_box.center[0]), str(final_box.center[1]), str(final_box.center[2]), str(final_box.wlh[0]),
                        #          str(final_box.wlh[1]), str(final_box.wlh[2]), str(final_box.orientation.radians)])
                        #     line_truth = ','.join(
                        #         [str(this_BB.center[0]), str(this_BB.center[1]), str(this_BB.center[2]), str(this_BB.wlh[0]),
                        #          str(this_BB.wlh[1]), str(this_BB.wlh[2]), str(this_BB.orientation.radians)])
                        #     f1.write(line_predict)
                        #     f1.write('\n')
                        #     f2.write(line_truth)
                        #     f2.write('\n')



                        if args.vis:
                            bat = d[str(i)]
                            bat_center = [-bat['center'][1], -bat['center'][2], bat['center'][0]]
                            bat_size = bat['size']
                            bat_rotation = Quaternion(axis=[0, 0, 1], radians=estimation_box[6]) * Quaternion(
                axis=[0, 1, 0], radians=np.pi / 2)

                            bat_box = Box(center=bat_center,
                                          size=bat_size,
                                          orientation=bat_rotation)
                            bat_center_box = copy.deepcopy(bat_box)
                            bat_center_box.translate(trans)
                            bat_center_box.rotate(Quaternion(matrix=(rot_mat)))

                            print(bat_center_box)
                            print(box)

                            pic_dir = os.path.join(f'{this_anno["type"]}_tracking_results', str(this_anno['scene']),
                                                   str(this_anno['track_id']))
                            if not os.path.exists(pic_dir):
                                os.makedirs(pic_dir)

                            fig = draw_lidar(candidate_PC, is_grid=False, is_axis=False,
                                             is_top_region=False)
                            draw_gt_boxes3d(gt_boxes3d=new_gt_box.corners().T.reshape(1, 8, 3), color=(0, 1, 0),
                                            fig=fig)
                            draw_gt_boxes3d(gt_boxes3d=box.corners().T.reshape(1, 8, 3), color=(1, 0, 0), fig=fig)
                            #draw_gt_boxes3d(gt_boxes3d=bat_center_box.corners().T.reshape(1, 8, 3), color=(0, 0, 1), fig=fig)
                            lidar_path = os.path.join(pic_dir, str(this_anno['frame']) + '_lidar.png')
                            # heat_path = os.path.join(pic_dir, str(this_anno['frame']) + '_heat2' + '.png')

                            mlab.savefig(lidar_path, figure=fig)
                            mlab.close()

                            proj, _ = utils.project_velo2rgb(this_BB, this_PJ_Matrix)
                            pproj, _ = utils.project_velo2rgb(final_box, this_PJ_Matrix)
                            #bproj, _ = utils.project_velo2rgb(bat_box, this_PJ_Matrix)
                            img = draw_rgb_projections(this_RGB, proj)
                            img = draw_rgb_projections(img, pproj, color=(255,0,0))
                            #img = draw_rgb_projections(img, bproj, color=(0,0,255))
                            rgb_path = os.path.join(pic_dir, str(this_anno['frame']) + '_rgb.png')
                            plt.imsave(rgb_path, img)

                        # box_2 = utils.getOffsetBB(ref_BB, estimate_offset_cpu)
                        # center_box_2 = utils.getOffsetBB(new_ref_box, estimate_offset_cpu)


                        # print(estimation_boxs_cpu[6])
                        # print(box.orientation.radians)
                        # print(new_this_box.orientation.radians)

                            # cv2.imwrite(heat_path, score_map)

                    # print(results_BBs[-1])
                    # print(BBs[i])

                    # estimate overlap/accuracy fro current sample

                    this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
                    this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)
                    results_success.append(this_overlap)
                    results_precision.append(this_accuracy)


                    Success_main.add_overlap(this_overlap)
                    Precision_main.add_accuracy(this_accuracy)
                    Success_batch.add_overlap(this_overlap)
                    Precision_batch.add_accuracy(this_accuracy)



                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    t.update(1)

                    if Success_main.count >= max_iter and max_iter >= 0:
                        return Success_main.average, Precision_main.average


                t.set_description('Test {}: '.format(epoch)+
                                  'Time {:.3f}s '.format(batch_time.avg)+
                                  '(it:{:.3f}s) '.format(batch_time.val)+
                                  'Data:{:.3f}s '.format(data_time.avg)+
                                  '(it:{:.3f}s), '.format(data_time.val)+
                                  'Succ/Prec:'+
                                  '{:.1f}/'.format(Success_main.average)+
                                  '{:.1f}'.format(Precision_main.average))
                logging.info(i)
                if i <= 50:
                    c_50 +=1
                    cl_50.append(Success_batch.average)
                    pl_50.append(Precision_batch.average)
                elif i > 50 and i <= 200:
                    c_100 += 1
                    cl_100.append(Success_batch.average)
                    pl_100.append(Precision_batch.average)
                else:
                    c_200 += 1
                    cl_200.append(Success_batch.average)
                    pl_200.append(Precision_batch.average)
                logging.info('batch {}'.format(batch_num) + 'Succ/Prec:' +
                             '{:.1f}/'.format(Success_batch.average) +
                             '{:.1f}'.format(Precision_batch.average))

                Success_batch.reset()
                Precision_batch.reset()
    logging.info('{},{},{}'.format(c_50, c_100, c_200))
    logging.info('{},{},{}'.format(sum(cl_50)/len(cl_50), sum(cl_100)/len(cl_100), sum(cl_200)/len(cl_200)))
    logging.info('{},{},{}'.format(sum(pl_50)/len(pl_50), sum(pl_100)/len(pl_100), sum(pl_200)/len(pl_200)))
    return Success_main.average, Precision_main.average


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--save_root_dir', type=str, default='./results_cyclist/',  help='output folder')
    parser.add_argument('--data_dir', type=str, default = '/home/lilium/zhuange/kitti/training',  help='dataset path')
    parser.add_argument('--model', type=str, default = 'cyclist.pth', help='model name for training resume')
    parser.add_argument('--category_name', type=str, default='Cyclist',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
    parser.add_argument('--shape_aggregation',required=False,type=str,default="firstandprevious",help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB',required=False,type=str,default="previous_result",help='previous_result/previous_gt/current_gt')
    parser.add_argument('--IoU_Space',required=False,type=int,default=3,help='IoUBox vs IoUBEV (2 vs 3)')
    parser.add_argument('--vis', type=bool, default=False, help='set to True if dumping visualization')
    args = parser.parse_args()
    print (args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(args.save_root_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log')), level=logging.INFO)
    logging.info('======================================================')

    args.manualSeed = 1
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    model = SiamPillar()
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, range(args.ngpu))
    if args.model != '':
        model.load_state_dict(torch.load(os.path.join(args.save_root_dir, args.model)), strict=False)
    model.cuda()
    print(model)
    torch.cuda.synchronize()



    # Car/Pedestrian/Van/Cyclist
    dataset_Test = SiameseTest(
            input_size=1024,
            path= args.data_dir,
            split='Test',
            category_name=args.category_name,
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

    if dataset_Test.isTiny():
        max_epoch = 2
    else:
        max_epoch = 1

    for epoch in range(max_epoch):
        Succ, Prec = test(
            test_loader,
            model,
            epoch=epoch + 1,
            shape_aggregation=args.shape_aggregation,
            reference_BB=args.reference_BB,
            IoU_Space=args.IoU_Space)
        Success_run.update(Succ)
        Precision_run.update(Prec)
        logging.info("mean Succ/Prec {}/{}".format(Success_run.avg,Precision_run.avg))
