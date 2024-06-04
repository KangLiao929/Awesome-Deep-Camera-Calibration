# Benchmark for Learning-based Camera Calibration and Beyond

As there is no public and unified benchmark in learning-based camera calibration, we contribute a dataset that can serve as a platform for generalization evaluations. In this dataset, the images and videos are captured by different cameras under diverse scenes, including simulation environments and real-world settings. Additionally, we provide the calibration ground truth, parameter label, and visual clues in this dataset based on different conditions. The directory structure of this benchmark is formed as follows.

```
├── Dataset
|   ├── Pinhole
|   ├── Distortion
|   ├── Cross-view
|   ├── Cross-sensor
```

Please feel free to use the whole benchmark from [[Google Dirve](https://drive.google.com/file/d/1ffNClmeFqQ_poKvSvYqu_JsTnzp6T6Ps/view?usp=sharing)]. If you only interest in one type of the camera model, please refer to each download link as follows.

## :circus_tent: Standard Pinhole Model [[Google Dirve](https://drive.google.com/file/d/11jNhxzx0WuQcrUlKzwGTf82DtH4viIoJ/view?usp=sharing)]

### Brief Description

We collected 300 high-resolution images on the Internet, captured by popular digital cameras such as Canon, Fujifilm, Nikon, Olympus, Sigma, Sony, etc. For each image, we provide the specific focal length of its lens. We have included a diverse range of subjects, including landscapes, portraits, wildlife, architecture, etc. The range of focal length is from 4.5mm to 600mm, allowing users to explore the different effects that different lenses can have on an image.

### Directory/Data Structure and Parsing
The value of the focal length of each image can be obtained by splitting the last sub-string in the file name (e.g., ```file_name.split("_")```).

```
├── Dataset
|   ├── Pinhole
|   |   ├── Canon EF-S 18-135mm F3.5-5.6 IS USM_18mm.jpg
|   |   ├── Canon EF-S 18-135mm F3.5-5.6 IS USM_24mm.jpg
|   |   ├── ......
```

## :circus_tent: Distortion Camera Model [[Google Dirve](https://drive.google.com/file/d/1qBGvAPOnuiG28uLde4vWxRYFuVZbQrDv/view?usp=sharing)]


### Brief Description

We created a comprehensive dataset for the distortion camera model, with a focus on wide-angle cameras. The dataset is comprised of three subcategories: (i) The first is a synthetic dataset, which was generated using the widely used polynomial model[1][2]. It contains both circular and rectangular structures, with 1,000 distortion-rectification image pairs. (ii) The second subcategory consists of data captured under real-world settings, derived from the raw calibration data for around 40 types of wide-angle cameras. For each calibration data, the intrinsics, extrinsics, and distortion coefficients are available. (iii) Finally, we exploit a car equipped with different cameras to capture video sequences. The scenes cover both indoor and outdoor environments, including daytime and nighttime footage.

### Directory/Data Structure and Parsing

For each folder of the 'Real' subcategory, the file name is formatted as 'lens-type_chip-type_resolution', such as '1002_2053_1080P', which contains the calibration results of a camera array with four wide-angle lenses. In particular, we provide the originally captured images of four wide-angle lenses in the 'lens-type_chip-type_icon' folder, such as '1002_2053_icon'. Besides, the calibration results of each wide-angle lens are included in the 'lens-type_chip-type_0', 'lens-type_chip-type_1', 'lens-type_chip-type_2', and 'lens-type_chip-type_3' folder, respectively:
* Rectified images in the 'calib_result' folder.
* Remapping table in the '.brp' file.
* Camera extrinsic parameters in the 'cam_mat_.xml' file.
* Camera distortion parameters in the 'dist_coeff_.xml' file.

For each folder of the 'Real_Sequence' subcategory, the equipment configuration is similar to the 'Real' subcategory. Besides the captured image ('0', '1', '2', and '3' represent 'front', 'rear', 'right', and 'left' equipped positions in the car), the remapping table, extrinsic parameters, and distortion parameters of each wide-angle lens, we also integrate the calibrated parameters of four lenses in the 'camera' folder ('.json').

```
├── Dataset
|   ├── Distortion
|   |   ├── Synthetic
|   |   |   ├── circular_structure
|   |   |   |   ├── input
|   |   |   |   ├── gt
|   |   |   ├── rectangular_structure
|   |   |   |   ├── input
|   |   |   |   ├── gt
|   |   ├── Real
|   |   |   ├── 1002_2053_1080P
|   |   |   |   ├── 1002_2053_0
|   |   |   |   |   ├── calib_result
|   |   |   |   |   ├── 1002_2053_0.brp
|   |   |   |   |   ├── cam_mat_1002_2053_0.xml
|   |   |   |   |   ├── dist_coeff_1002_2053_0.xml
|   |   |   |   ├── ......
|   |   |   |   ├── 1002_2053_3
|   |   |   |   |   ├── ......
|   |   |   |   ├── 1002_2053_icon
|   |   |   |   |   ├── 0
|   |   |   |   |   ├── ......
|   |   |   |   |   ├── 3
|   |   ├── Real_Sequence
|   |   |   ├── data_2022_04_06
|   |   |   |   ├── camera
|   |   |   |   |   ├── camera matrix
|   |   |   |   |   |   ├── 6053_1335_0.brp
|   |   |   |   |   |   ├── ......
|   |   |   |   |   |   ├── 6053_1335_3.brp
|   |   |   |   |   |   ├── cam_mat_6053_1335_0.xml
|   |   |   |   |   |   ├── ......
|   |   |   |   |   |   ├── cam_mat_6053_1335_3.xml
|   |   |   |   |   |   ├── dist_coeff_6053_1335_0.xml
|   |   |   |   |   |   ├── ......
|   |   |   |   |   |   ├── dist_coeff_6053_1335_3.xml
|   |   |   |   |   ├── camera_2022_04_06.json
|   |   |   |   ├── sequences
|   |   |   |   |   ├── 001
|   |   |   |   |   |   ├── jpg
|   |   |   |   |   ├── ......
```

### Camera Parameter Parsing for 'Real_Sequence' Subcategory

We provide the calibrated intrinsic and extrinsic parameters of four cameras equipped in a car, which can be found in '.\camera\camera_x_x_x.json'. Especially for the extrinsic parameters, the 'position' denotes the 3D coordinates (XYZ) of each camera with respect to the origin, the point where the midpoint of the car's front axle is projected onto the ground. The X-axis points to the left side of the car, the Y-axis points to the direction of travel, and the Z-axis points upward from the ground. Moreover, the 'pose' represents the Euler angles of each camera, i.e., roll, pitch, and yaw.


## :circus_tent: Cross-View Model [[Google Dirve](https://drive.google.com/file/d/16xUU3hAvRv6DnEZ126TI0zSKRZxNXy08/view?usp=sharing)]

### Brief Description

We selected 500 testing samples at random from each of the four representative datasets[3][4][5] to create a dataset for the cross-view model. It covers a range of scenarios: MS-COCO provides natural synthetic data, GoogleEarch contains aerial synthetic data, and GoogleMap offers multi-modal synthetic data. Parallax is not a factor in these three datasets, while CAHomo provides real-world data with non-planar scenes. To standardize the dataset, we converted all images to a unified format and recorded the matched points between two views. In MS-COCO, GoogleEarch, and GoogleMap, we used four vertices of the images as the matched points. In CAHomo, we identified six matched key points within the same plane.

### Directory/Data Structure and Parsing

We unified the format of all datasets as follows. In the label, we record the matched points between img1 and img2. In MSCOCO, GoogleEarch, and GoogleMap, we adopt the four vertices while in CAHomo, we leverage six matched key points induced in the same plane. For LK-based alignment algorithms, we also provide the original images of img2 (named 'img2_ori') in the datasets of MSCOCO, GoogleEarch, and GoogleMap.

```
├── Dataset
|   ├── Cross-view
|   |   ├── MSCOCO
|   |   |   ├── img1
|   |   |   ├── img2
|   |   |   ├── img2_ori
|   |   |   ├── label
|   |   ├── GoogleEarth
|   |   |   ├── img1
|   |   |   ├── img2
|   |   |   ├── img2_ori
|   |   |   ├── label
|   |   ├── GoogleMap
|   |   |   ├── img1
|   |   |   ├── img2
|   |   |   ├── img2_ori
|   |   |   ├── label
|   |   ├── CAHomo
|   |   |   ├── img1
|   |   |   ├── img2
|   |   |   ├── label
|   |   |   ├── visualization
```

## :circus_tent: Cross-Sensor Model [[Google Dirve](https://drive.google.com/file/d/1DgPyUTqDwjl95Rs0XMNXBkzbCU4ocBtG/view?usp=sharing)]

### Brief Description

We collected RGB and point cloud data from Apollo[6], DAIR-V2X[7], KITTI[8], KUCL[9], NuScenes[10], and ONCE[11]. Around 300 data pairs with calibration parameters are included in each category. The datasets are captured in different countries to provide enough variety. Each dataset has a different sensor setup, obtaining camera-LiDAR data with varying image resolution, LiDAR scan pattern, and camera-LiDAR relative location. The image resolution ranges from 2448x2048 to 1242x375, while the LiDAR sensors are from Velodyne and Hesai, with 16, 32, 40, 64, and 128 beams. They include not only normal surrounding multi-view images but also small baseline multi-view data. Additionally, we also added random disturbance of around 20 degrees rotation and 1.5 meters translation based on classical settings to simulate vibration and collision.

### Directory/Data Structure and Parsing

In the context of Camera-LiDAR calibration, current learning-based methods typically necessitate an RGB image with an accompanying point cloud or depth map as input. For ease of use, we organize the input images, point clouds, depth maps, and ground-truth (GT) labels into four folders that share a consistent structure. The intrinsic parameters and distortion coefficients are stored in the 'params' folder. Additionally, we provide pseudo-color visualizations of depth maps overlaid on RGB images within the 'visualization' folder for preview purposes.

Here is a detailed illustration of the dataset:
* RGB images have been pre-processed to remove distortion.
* Point clouds are transformed using the SE(3) labels. 
* Depth maps are projected using intrinsic parameters, distortion coefficients, and GT labels, then saved in uint16 format representing millimeters.
* GT labels are formatted as 4x4 matrices.
* Each RGB image, point cloud, depth map, and GT label shares the same filename for clear correspondence.

```
├── Dataset
|   ├── Cross-sensor
|   |   ├── image
|   |   |   ├── Apollo
|   |   |   |   ├── front_image
|   |   |   |   ├── left_back_image
|   |   |   |   ├── right_back_image
|   |   |   ├── DAIR-V2X
|   |   |   ├── KITTI
|   |   |   |   ├── image_02
|   |   |   |   ├── image_03
|   |   |   ├── KUCL
|   |   |   |   ├── cam0
|   |   |   |   ├── cam1
|   |   |   |   ├── ......
|   |   |   |   ├── cam5
|   |   |   |   ├── mask
|   |   |   ├── NUScenes
|   |   |   |   ├── CAM_BACK
|   |   |   |   ├── ......
|   |   |   |   ├── CAM_FRONT
|   |   |   ├── ONCE
|   |   |   |   ├── val_cam01
|   |   |   |   ├── val_cam03
|   |   |   |   ├── ......
|   |   ├── depth
|   |   |   ├── Apollo
|   |   |   |   ├── ......
|   |   |   ├── DAIR-V2X
|   |   |   ├── KITTI
|   |   |   |   ├── ......
|   |   |   ├── KUCL
|   |   |   |   ├── ......
|   |   |   ├── NUScenes
|   |   |   |   ├── ......
|   |   |   ├── ONCE
|   |   |   |   ├── ......
|   |   ├── pcd
|   |   |   ├── ......
|   |   ├── label
|   |   |   ├── ......
|   |   ├── visualization
|   |   |   ├── ......
|   |   ├── params
|   |   |   ├── ......
```



## Reference
```markdown
[1] Xiaoqing Yin, Xinchao Wang, Jun Yu, Maojun Zhang, Pascal Fua, and Dacheng Tao. "Fisheyerecnet: A multi-context collaborative deep network for fisheye image rectification." European Conference on Computer Vision (ECCV), 2018.
[2] Kang Liao, Chunyu Lin, Yao Zhao, and Moncef Gabbouj. "DR-GAN: Automatic radial distortion rectification using conditional GAN in real-time." IEEE Transactions on Circuits and Systems for Video Technology, 2019.
[3] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Deep image homography estimation,” arXiv preprint arXiv:1606.03798, 2016.
[4] Y. Zhao, X. Huang, and Z. Zhang, “Deep lucas-kanade homography for multimodal image alignment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
[5] J. Zhang, C. Wang, S. Liu, L. Jia, N. Ye, J. Wang, J. Zhou, and J. Sun, “Content-aware unsupervised deep homography estimation,” in European Conference on Computer Vision (ECCV), 2020.
[6] X. Huang, P. Wang, X. Cheng, D. Zhou, Q. Geng, and R. Yang, “The apolloscape open dataset for autonomous driving and its application,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.
[7] H. Yu, Y. Luo, M. Shu, Y. Huo, Z. Yang, Y. Shi, Z. Guo, H. Li, X. Hu, J. Yuan et al., “Dair-v2x: A large-scale dataset for vehicle infrastructure cooperative 3d object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
[8] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
[9] J. Kang and N. L. Doh, “Automatic targetless camera–LIDAR calibration by aligning edge with Gaussian mixture model,” Journal of Field Robotics, 2020.
[10] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
[11] Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang, Yamin Li et al. "One million scenes for autonomous driving: Once dataset." arXiv preprint arXiv:2106.11037, 2021.

```
