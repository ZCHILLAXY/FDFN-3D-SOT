import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mayavi import mlab
from torch.utils.data import Dataset

from config import cfg
from utils.draw_utils import draw_lidar, draw_gt_boxes3d, draw_rgb_projections

from utils.data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import utils.common_utils as utils
from utils.searchspace import KalmanFiltering
import logging

from utils.spconv_preprocess import spconv_process_pointcloud, spconv_process_project


class kittiDataset():

    def __init__(self, path):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.KITTI_rgb = os.path.join(self.KITTI_Folder, "image_02")

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [3]
            else:
                sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        # print(self.list_of_scene)
        list_of_tracklet_anno = []
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index,
                                 anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)
        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '{:06}.bin'.format(box["frame"]))
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)
        except:
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    if values[0] == 'R_rect':
                        data[values[0]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 3)
                    else:
                        data[values[0]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data


class SiameseDataset(Dataset):

    def __init__(self,
                 input_size,
                 path,
                 split,
                 category_name="Car",
                 offset_BB=0,
                 scale_BB=1.0):

        self.dataset = kittiDataset(path=path)

        self.input_size = input_size
        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPCandRGB = self.dataset.getBBandPC

        self.category_name = category_name

        self.list_of_tracklet_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name)
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class SiameseTrain(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,


                 split="",
                 category_name="Car",
                 offset_BB=0,
                 scale_BB=1.0):
        super(SiameseTrain, self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            offset_BB=offset_BB,
            scale_BB=scale_BB)

        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

        self.num_candidates_perframe = 4

        logging.info("preloading PC...")
        self.list_of_PCs = [None] * len(self.list_of_anno)
        self.list_of_BBs = [None] * len(self.list_of_anno)
        self.list_of_Calibs = [None] * len(self.list_of_anno)


        logging.info("preloading Boxes..")
        for index in tqdm(range(len(self.list_of_anno))):
            anno = self.list_of_anno[index]
            calib_path = os.path.join(self.dataset.KITTI_Folder, 'calib',
                                      anno['scene'] + ".txt")
            calib = self.dataset.read_calib_file(calib_path)
            PC, box = self.getBBandPCandRGB(anno)
            new_PC = utils.cropPC(PC, box, offset=10)
            # gt_3dTo2D = utils.project_velo2rgb(box, calib)
            # img_with_box = draw_rgb_projections(RGB, gt_3dTo2D, color=(0, 255, 0), thickness=1)
            # plt.imshow(new_RGB)
            # plt.show()
            self.list_of_PCs[index] = new_PC
            self.list_of_BBs[index] = box
            self.list_of_Calibs[index] = calib

        logging.info("Boxes preloaded!")

        logging.info("preloading Model Index..")
        self.model_PC = [None] * len(self.list_of_tracklet_anno)
        for i in tqdm(range(len(self.list_of_tracklet_anno))):
            list_of_anno = self.list_of_tracklet_anno[i]
            # PCs = []
            # BBs = []
            cnt = 0
            for anno in list_of_anno:
                # this_PC, this_BB = self.getBBandPC(anno)
                # PCs.append(this_PC)
                # BBs.append(this_BB)
                anno["model_idx"] = i
                anno["relative_idx"] = cnt
                cnt += 1

            # self.model_PC[i] = utils.getModel(
            #     PCs, BBs, offset=self.offset_BB, scale=self.scale_BB)

        logging.info("Model Index preloaded!")

    def __getitem__(self, index):
        return self.getitem(index)

    def getDatafromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        this_Calib = self.list_of_Calibs[anno_idx]
        return this_PC, this_BB, this_Calib

    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(3)
        else:
            gaussian = KalmanFiltering(bnd=[0.5, 0.5, 5])
            sample_offsets = gaussian.sample(1)[0]

        this_anno = self.list_of_anno[anno_idx]

        this_PC, this_BB, this_Calib = self.getDatafromIndex(anno_idx)
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)
        sample_pjmatrix = utils.getPJMatrix(this_Calib)

        # sample_PC = utils.cropAndCenterPC(
        #      this_PC, sample_BB, offset=self.offset_BB, scale=self.scale_BB)
        # print(sample_PC)
        sample_PC, sample_box, sample_trans, sample_rotmat = utils.cropAndCenterPC_new(
            this_PC, sample_BB, this_BB, offset=self.offset_BB, scale=self.scale_BB)

        this_rgb_path = os.path.join(self.dataset.KITTI_rgb, this_anno["scene"],
                                         '{:06}.png'.format(this_anno["frame"]))
        this_RGB = mpimg.imread(this_rgb_path)

        sample_RGB = utils.cropRGB_train(this_RGB, sample_BB, sample_pjmatrix, search=True, offset=self.offset_BB, scale=self.scale_BB)
        sample_forward_PC = utils.cropForwardPC(this_PC, this_RGB, sample_RGB, sample_BB, sample_pjmatrix, search=True, offset=self.offset_BB, scale=self.scale_BB)


        if sample_PC.nbr_points() < 10 or sample_RGB.shape[0] == 0 or sample_RGB.shape[1] == 0:
            return self.getitem(np.random.randint(0, self.__len__()))
        sample_PC = utils.regularizePC_scene(sample_PC, self.input_size)
        target_center = np.expand_dims(sample_box[0:3], axis=0)
        sample_PC = np.concatenate([sample_PC, target_center], axis=0)

        sample_forward_PC = utils.regularizePC_scene(sample_forward_PC, self.input_size) # (1024,3)
        sample_RGB = cv2.resize(sample_RGB, (cfg.SCENE_INPUT_WIDTH, cfg.SCENE_INPUT_WIDTH), cv2.INTER_NEAREST)

        # u, v, z = sample_forward_PC.T
        # # print(u.max())
        # # print(z)
        # # print(sample_forward_PC.shape)
        # plt.imshow(sample_RGB)
        # plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
        # plt.show()

        if this_anno["relative_idx"] == 0:
            prev_idx = 0
            fir_idx = 0
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        gt_PC_pre, gt_BB_pre, gt_Calib_pre = self.getDatafromIndex(prev_idx)
        gt_PC_fir, gt_BB_fir, _ = self.getDatafromIndex(fir_idx)
        gt_pj_matrix = utils.getPJMatrix(gt_Calib_pre)


        if sample_idx == 0:
            samplegt_offsets = np.zeros(3)
        else:
            samplegt_offsets = np.random.uniform(low=-0.1, high=0.1, size=3)
            samplegt_offsets[2] = samplegt_offsets[2]*5
        gt_BB_pre = utils.getOffsetBB(gt_BB_pre, samplegt_offsets)

        gt_PC = utils.getModel([gt_PC_fir, gt_PC_pre], [
                               gt_BB_fir, gt_BB_pre], offset=self.offset_BB, scale=self.scale_BB)
        prev_anno = self.list_of_anno[prev_idx]
        prev_rgb_path = os.path.join(self.dataset.KITTI_rgb, prev_anno["scene"],
                                '{:06}.png'.format(prev_anno["frame"]))
        prev_RGB = mpimg.imread(prev_rgb_path)
        gt_RGB = utils.cropRGB_train(prev_RGB, gt_BB_pre, gt_pj_matrix)
        gt_forward_PC = utils.cropForwardPC(gt_PC_pre, prev_RGB, gt_RGB, gt_BB_pre, gt_pj_matrix)


        if gt_PC.nbr_points() < 1 or gt_RGB.shape[0] == 0 or gt_RGB.shape[1] == 0:
            return self.getitem(np.random.randint(0, self.__len__()))

        gt_PC = utils.regularizePC_template(gt_PC, self.input_size)
        object_center = np.zeros((1, 3))
        gt_PC = np.concatenate([gt_PC, object_center], axis=0)

        gt_forward_PC = utils.regularizePC_template(gt_forward_PC, self.input_size)
        gt_RGB = cv2.resize(gt_RGB, (cfg.TEMPLATE_INPUT_WIDTH, cfg.TEMPLATE_INPUT_WIDTH), cv2.INTER_NEAREST)

        # u, v, z = gt_forward_PC.T
        # plt.imshow(gt_RGB)
        # plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
        # plt.show()

        template_voxel_dict = spconv_process_pointcloud(gt_PC, template=True)
        scene_voxel_dict = spconv_process_pointcloud(sample_PC)

        rgb_template_voxel_dict = spconv_process_project(gt_forward_PC, template=True)
        rgb_scene_voxel_dict = spconv_process_project(sample_forward_PC)

        s_vox_feature = scene_voxel_dict['feature_buffer']
        s_vox_number = scene_voxel_dict['number_buffer']
        s_vox_coordinate = scene_voxel_dict['coordinate_buffer']

        t_vox_feature = template_voxel_dict['feature_buffer']
        t_vox_number = template_voxel_dict['number_buffer']
        t_vox_coordinate = template_voxel_dict['coordinate_buffer']

        rgb_t_vox_feature = rgb_template_voxel_dict['feature_buffer']
        rgb_t_vox_number = rgb_template_voxel_dict['number_buffer']
        rgb_t_vox_coordinate = rgb_template_voxel_dict['coordinate_buffer']
        rgb_s_vox_feature = rgb_scene_voxel_dict['feature_buffer']
        rgb_s_vox_number = rgb_scene_voxel_dict['number_buffer']
        rgb_s_vox_coordinate = rgb_scene_voxel_dict['coordinate_buffer']

        # b_center = [sample_box[0], sample_box[1], sample_box[2]]
        # b_size = [sample_box[3], sample_box[4], sample_box[5]]
        # b_rot = Quaternion(axis=[0, 0, 1], radians=sample_box[6])
        # box = Box(center=b_center,
        #           size=b_size,
        #           orientation=b_rot)
        # fig = draw_lidar(sample_PC, is_grid=False, is_axis=False,
        #                  is_top_region=False)
        # draw_gt_boxes3d(gt_boxes3d=box.corners().T.reshape(1, 8, 3), color=(0, 1, 0), fig=fig)
        # mlab.show()


        # a,b = utils.project_velo2rgb(sample_BB, sample_pjmatrix)
        # img = draw_rgb_projections(this_RGB, a)
        # plt.imshow(img)
        # # plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
        # plt.show()

        # b_center = sample_box[0:3]
        # b_size = sample_box[3:6]
        # b_rot = Quaternion(axis=[0, 0, 1], radians=sample_box[6])
        #
        # box = Box(center=b_center,
        #           size=b_size,
        #           orientation=b_rot)
        #
        # box.rotate(Quaternion(matrix=(np.transpose(sample_rotmat))))
        # box.translate(-sample_trans)
        # print(box)
        # box_projection, box_rgb = utils.project_velo2rgb(box, sample_pjmatrix)
        # print(box_rgb)
        # #
        # qqqqq = sample_RGB[int(32 - (box_rgb[3] - box_rgb[1]) / 2): int(32 + (box_rgb[3] - box_rgb[1]) / 2), int(32 - (box_rgb[2] - box_rgb[0]) / 2): int(32 + (box_rgb[2] - box_rgb[0]) / 2)]
        # # plt.imshow(ppppp)
        # # plt.show()
        # plt.imshow(qqqqq)
        # plt.show()

        gt_box_lst = sample_box
        template_box = np.expand_dims(np.array([0, 0, 0, sample_box[3], sample_box[4], sample_box[5], 0]), 0).repeat(t_vox_feature.shape[0], axis=0)
        sample_box = np.expand_dims(np.array(sample_box), 0).repeat(s_vox_feature.shape[0], axis=0)

        return t_vox_feature, t_vox_number, t_vox_coordinate, \
               s_vox_feature, s_vox_number, s_vox_coordinate, \
               rgb_t_vox_feature, rgb_t_vox_number, rgb_t_vox_coordinate, \
               rgb_s_vox_feature, rgb_s_vox_number, rgb_s_vox_coordinate, \
               gt_RGB, sample_RGB, \
               template_box, sample_box, gt_box_lst

    def __len__(self):
        nb_anno = len(self.list_of_anno)
        return nb_anno * self.num_candidates_perframe

    def getAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def getSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)


class SiameseTest(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 offset_BB=0,
                 scale_BB=1.0):
        super(SiameseTest, self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            offset_BB=offset_BB,
            scale_BB=scale_BB)
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        RGBs = []
        BBs = []
        Calibs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPCandRGB(anno)

            this_rgb_path = os.path.join(self.dataset.KITTI_rgb, anno["scene"],
                                         '{:06}.png'.format(anno["frame"]))
            this_BGR = cv2.imread(this_rgb_path)
            this_RGB = cv2.cvtColor(this_BGR, cv2.COLOR_BGR2RGB)

            calib_path = os.path.join(self.dataset.KITTI_Folder, 'calib',
                                      anno['scene'] + ".txt")
            calib = self.dataset.read_calib_file(calib_path)

            PCs.append(this_PC)
            RGBs.append(this_RGB)
            BBs.append(this_BB)
            Calibs.append(calib)

        return PCs, RGBs, BBs, Calibs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)


if __name__ == '__main__':

    dataset_Training = SiameseTrain(
        input_size=1024,
        path='/home/lilium/zhuange/kitti/training',
        split='Train_tiny',
        category_name='Car',
        offset_BB=0,
        scale_BB=1.25)

    aa = dataset_Training.getitem(20)





