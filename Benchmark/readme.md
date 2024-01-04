# Benchmark for Learning-based Camera Calibration and Beyond

## :circus_tent: Standard Pinhole Model 

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

## :circus_tent: Distortion Camera Model 

### Brief Description
We created a comprehensive dataset for the distortion camera model, with a focus on wide-angle cameras. The dataset is comprised of three subcategories. The first is a synthetic dataset, which was generated using the widely used polynomial model[1][2]. It contains both circular and rectangular structures, with 1,000 distortion-rectification image pairs. The second subcategory consists of data captured under real-world settings, derived from the raw calibration data for around 40 types of wide-angle cameras. For each calibration data, the intrinsics, extrinsics, and distortion coefficients are available. Finally, we exploit a car equipped with different cameras to capture video sequences. The scenes cover both indoor and outdoor environments, including daytime and nighttime footage.

### Directory/Data Structure and Parsing



## :circus_tent: Cross-View Model

### Brief Description

We selected 500 testing samples at random from each of the four representative datasets[3][4][5] to create a dataset for the cross-view model. It covers a range of scenarios: MS-COCO provides natural synthetic data, GoogleEarch contains aerial synthetic data, and GoogleMap offers multi-modal synthetic data. Parallax is not a factor in these three datasets, while CAHomo provides real-world data with non-planar scenes. To standardize the dataset, we converted all images to a unified format and recorded the matched points between two views. In MS-COCO, GoogleEarch, and GoogleMap, we used four vertices of the images as the matched points. In CAHomo, we identified six matched key points within the same plane.

### Directory/Data Structure and Parsing

We unified the format of all datasets as follows. In the label, we record the matched points between img1 and img2. In MSCOCO, GoogleEarch, and GoogleMap, we adopt the four vertices while in CAHomo, we leverage six matched key points induced in the same plane. For LK-based alignment algorithms, we also provide the original images of img2 in the datasets of MSCOCO, GoogleEarch, and GoogleMap.

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

## :circus_tent: Cross-Sensor Model

### Brief Description

We collected RGB and point cloud data from Apollo[6], DAIR-V2X[7], KITTI[8], KUCL[9], NuScenes[10], and ONCE[11]. Around 300 data pairs with calibration parameters are included in each category. The datasets are captured in different countries to provide enough variety. Each dataset has a different sensor setup, obtaining camera-LiDAR data with varying image resolution, LiDAR scan pattern, and camera-LiDAR relative location. The image resolution ranges from 2448x2048 to 1242x375, while the LiDAR sensors are from Velodyne and Hesai, with 16, 32, 40, 64, and 128 beams. They include not only normal surrounding multi-view images but also small baseline multi-view data. Additionally, we also added random disturbance of around 20 degrees rotation and 1.5 meters translation based on classical settings to simulate vibration and collision.

### Directory/Data Structure and Parsing

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
