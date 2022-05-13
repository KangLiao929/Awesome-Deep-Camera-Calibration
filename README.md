# Awesome-Deep-Camera-Calibration

## [Deep Learning for Camera Calibration: A Survey]()

This repository provides：

 **1)** a unified online platform
 **2)** a new dataset 
 **3)** collects deep learning-based camera calibration **methods, datasets, and evaluation metrics**. 

More content and details can be found in our Survey Paper: [Deep Learning for Camera Calibration: A Survey](). 



## 📣News
1. The survey is submitted. 



## 🌱Contents
1. [Basics](#Basics)
2. [Methods](#Methods)
3. [Datasets](#Datasets)
4. [Metrics](#Metrics)
5. [Citation](#Citation)

## 📋Basics
* [Multiple view geometry in computer vision](https://cseweb.ucsd.edu/classes/sp13/cse252B-a/HZ2eCh2.pdf) - Hartley, R., & Zisserman, A. (2004)

## 📋Methods
![Overview](/chronology.png)
|Year|Publication|Title|Abbreviation|Objective|Platform|Network|
|---|---|---|---|---|---|---|
|2015|ICIP|Deepfocal: A method for direct focal length estimation [paper](https://ieeexplore.ieee.org/abstract/document/7351024)|DeepFocal|Intrinsics|Caffe|AlexNet|
|2015|ICCV|Posenet: A convolutional network for real-time 6-dof camera relocalization [paper](https://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html)|PoseNet|Extrinsics|Caffe|GoogLeNet|
|2016|BMVC|Horizon lines in the wild [paper](https://arxiv.org/abs/1604.02129)|DeepHorizon|Extrinsics|Caffe|GoogLeNet|
|2016|CVPR|Detecting vanishing points using global image context in a non-manhattan world [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhai_Detecting_Vanishing_Points_CVPR_2016_paper.html)|Deep VP|Extrinsics|Caffe|AlexNet|
|2016|ACCV|Radial lens distortion correction using convolutional neural networks trained with synthesized images [paper](https://link.springer.com/chapter/10.1007/978-3-319-54187-7_3)|Deep VP|Distortion coefficients|Caffe|AlexNet|
|2016|RSSW|Deep image homography estimation [paper](https://arxiv.org/abs/1606.03798)|Deep Homo|Projection matrixs|Caffe|VGG|
|2017|CVPR|Clkn: Cascaded lucas-kanade networks for image alignment [paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.html)|CLKN|Projection matrixs|Torch|CNN + Lucas-Kanade layer|
|2018|CVPR|A perceptual measure for deep single image camera calibration [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hold-Geoffroy_A_Perceptual_Measure_CVPR_2018_paper.html)|Hold-Geoffroy et al.|Intrinsics + Extrinsics| |DenseNet|
|2018|CVMP|DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras [paper](https://dl.acm.org/doi/abs/10.1145/3278471.3278479)|DeepCalib|Intrinsics + Distortion coefficients|TensorFlow|Inception-V3|
|2018|ECCV|Fisheyerecnet: A multi-context collaborative deep network for fisheye image rectification [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiaoqing_Yin_FishEyeRecNet_A_Multi-Context_ECCV_2018_paper.html)|FishEyeRecNet|Distortion coefficients|Caffe|VGG|
|2018|ICPR|Radial lens distortion correction by adding a weight layer with inverted foveal models to convolutional neural networks [paper](https://ieeexplore.ieee.org/abstract/document/8545218)|Shi et al.|Distortion coefficients|PyTorch|ResNet|
|2018|ECCV|Deep fundamental matrix estimation [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.html)|DeepFM|Projection matrixs|PyTorch|ResNet|
|2018|ECCVW|Deep fundamental matrix estimation without correspondences [paper](https://openaccess.thecvf.com/content_eccv_2018_workshops/w16/html/Poursaeed_Deep_Fundamental_Matrix_Estimation_without_Correspondences_ECCVW_2018_paper.html)|Poursaeed et al.|Projection matrixs| |CNNs|
|2018|RAL|Unsupervised deep homography: A fast and robust homography estimation model [paper](https://ieeexplore.ieee.org/abstract/document/8302515)|UDH|Projection matrixs|TensorFlow|VGG|
|2019|CVPR|Deep single image camera calibration with radial distortion [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Lopez_Deep_Single_Image_Camera_Calibration_With_Radial_Distortion_CVPR_2019_paper.html)|Lopez et al.|Intrinsics + Extrinsics + Distortion coefficients|PyTorch|DenseNet|





## 📋Datasets
|Abbreviation|Number|Format|Real/Synetic|Video|Paired/Unpaired/Application|Dataset|
|---|---|---|---|---|---|---|
|LOL [paper](https://arxiv.org/abs/1808.04560)|500|RGB|Real|No|Paired|[Dataset](https://daooshee.github.io/BMVC2018website/)|



## 📋Metrics
|Abbreviation|Full-/Non-Reference|Platform|Code|
|---|---|---|---|
|MAE (Mean Absolute Error)|Full-Reference| | |
|MSE (Mean Square Error)|Full-Reference| | |
|PSNR (Peak Signal-to-Noise Ratio)|Full-Reference| | |
|SSIM (Structural Similarity Index Measurement)|Full-Reference|MATLAB|[Code](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) |
|LPIPS (Learned Perceptual Image Patch Similarity)|Full-Reference|PyTorch|[Code](https://github.com/richzhang/PerceptualSimilarity) |
|LOE (Lightness Order Error)|Non-Reference|MATLAB|[Code](https://drive.google.com/drive/folders/0B3YzCh6G4aubLUhQMzdzR05nSDg?usp=sharing) |
|NIQE (Naturalness Image Quality Evaluator)|Non-Reference|MATLAB|[Code](https://github.com/utlive/niqe)|
|PI (Perceptual Index)|Non-Reference|MATLAB|[Code](https://github.com/chaoma99/sr-metric)|
|SPAQ (Smartphone Photography Attribute and Quality)|Non-Reference|PyTorch|[Code](https://github.com/h4nwei/SPAQ)|
|NIMA (Neural Image Assessment)|Non-Reference|PyTorch/TensorFlow|[Code](https://github.com/kentsyx/Neural-IMage-Assessment)/[Code](https://github.com/titu1994/neural-image-assessment)|
|MUSIQ (Multi-scale Image Quality Transformer)|Non-Reference|TensorFlow|[Code](https://github.com/google-research/google-research/tree/master/musiq)|
## 📜</g-emoji>License
The code, platform, and dataset are made available for academic research purpose only. 

## 📚</g-emoji>Citation
If you find the repository helpful in your resarch, please cite the following paper.
```
@article{LoLi,
  title={Low-Light Image and Video Enhancement Using Deep Learning: A Survey},
  author={Li, Chongyi and Guo, Chunle and Han, Linghao and Jiang, Jun and Cheng, Ming-Ming and Gu, Jinwei and Loy, Chen Change},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
## 📋Paper
[Official Version](https://ieeexplore.ieee.org/document/9609683)

[arXiv Version](https://arxiv.org/pdf/2104.10729.pdf)


## 📭Contact

```
kang_liao@bjtu.edu.cn
```
