#!/usr/bin/python3
"""
@Project: SiamPillar 
@File: draw_utils.py
@Author: Zhuang Yi
@Date: 2020/8/28
"""
import cv2
from mayavi import mlab
import numpy as np
from config import cfg
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt



def draw_lidar(lidar, is_grid=False, is_axis = False, is_top_region=False, fig=None, bsize=None):

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    #prs=lidar[:,3]

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs,
        colormap='spectral',  #'bone',  #'spectral',  #'copper',
        scale_factor=0.07,
        figure=fig)

    #draw grid
    if is_grid:
        w = bsize[0]
        l = bsize[1]
        h = bsize[2]
        print(bsize)
        mlab.points3d(0, 0, 0, color=(1, 0, 0), mode='sphere', scale_factor=0.2)
        x_min = -w / 2
        dx = 0.16
        z_min = -h / 2
        dz = 0.16
        kwargs = {'color': (0.5, 0.5, 1), 'tube_radius':None, 'line_width':3, 'figure':fig}
        mlab.points3d(0, x_min+0.08, z_min+0.08, color=(0, 0, 1), mode='sphere', scale_factor=0.1)
        for i in range(0, 2):
            for j in range(0, 2):
                xx = [x_min, x_min, x_min + dx * i, x_min + dx * i]
                zz = [z_min, z_min + dz * j, z_min + dz * j, z_min]
                mlab.plot3d([-3.2]*4 , xx, zz, **kwargs)
                mlab.plot3d([3.2]*4, xx, zz, **kwargs)
                mlab.plot3d([-3.2, 3.2], [x_min + dx * i, x_min + dx * i], [z_min + dz * j, z_min + dz * j],
                              **kwargs)
                mlab.plot3d([-3.2, 3.2], [x_min + dx * j, x_min + dx * j], [z_min + dz * i, z_min + dz * i],
                              **kwargs)


    #draw axis
    if is_axis:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = cfg.SCENE_X_MIN
        x2 = cfg.SCENE_X_MAX
        y1 = cfg.SCENE_Y_MIN
        y2 = cfg.SCENE_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=120, elevation=None, distance=30, focalpoint=[2.0909996, -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,0,0), line_width=2) -> object:


    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        for k in range(0,4):

            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+3)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=160, elevation=None, distance=20, focalpoint=[2.0909996, -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

def draw_rgb_projections(image, projections, color=(0, 255, 0), thickness=1, darker=1):

    img = image.copy()*darker
    forward_color = (0, 255, 0)
    qs = np.array(projections, dtype=int)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        # cv2.line(img, (qs[0,0],qs[0,1]), (qs[1,0],qs[1,1]), forward_color, thickness, cv2.LINE_AA)
        # cv2.line(img, (qs[1,0],qs[1,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        # cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        # cv2.line(img, (qs[3,0],qs[3,1]), (qs[0,0],qs[0,1]), forward_color, thickness, cv2.LINE_AA)
        # cv2.line(img, (qs[3,0],qs[3,1]), (qs[1,0],qs[1,1]), forward_color, thickness, cv2.LINE_AA)
        # cv2.line(img, (qs[2,0],qs[2,1]), (qs[0,0],qs[0,1]), forward_color, thickness, cv2.LINE_AA)

    return img

def draw_pillar_pc(lidar):
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(pxs, pys, pzs, c=(0.5, 0.2, 0.07))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.add_axes(ax)
    return fig

