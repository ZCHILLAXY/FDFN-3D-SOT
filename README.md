# FDFN-3D-SOT
Fusion-enhanced Feature Deblurring for 3D Single Object Tracking (FDFN)

## Introduction

We exploit both point cloud and RGB image features for the target and the search area. Substituting for the common correlation, the proposed feature deblurring fusion network (FDFN) enables an effective way in 3D SOT. 

## Preliminary

* Minimum Dependencies(Main):

python==3.7  
torch==1.7.0  
spconv==1.0(https://github.com/traveller59/spconv)  
DCN_v2 (https://github.com/jinfagang/DCNv2_latest)  
PS-ViT (https://github.com/yuexy/PS-ViT)
prroi_pooling (https://github.com/vacancy/PreciseRoIPooling)
mayavi==4.7.1  
opencv-python==4.3.0.36  
pyquarternion==0.9.5  
Shapely==1.7.0  

* Build box_overlap module in utils
```
    python setup.py
```

* Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

	Download [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) in the dataset and place them under the same parent folder.

## Training

Train a new FDFN model for one category
```
python train_tracking.py --data_dir=<kitti data path> --category_name=<category name>
```

Or, you can first change specific parameters and directly run
```
python train_tracking.py
```

## Testing
Test a new FDFN model, just do similarly as mentioned in Training process
```
python test_tracking.py --data_dir=<kitti data path> --category_name=<category name>
```

Please refer to the article for other settings.


## Acknowledgements

Thank Giancola for his implementation of [SC3D](https://github.com/SilvioGiancola/ShapeCompletion3DTracking). 
Thank Qi for his implementation of [P2B](https://github.com/HaozheQi/P2B). 
Thank Zheng for his implementation of [BAT](https://github.com/Ghostish/BAT). 
