# Awesome-Deep-Camera-Calibration
[![arXiv](https://img.shields.io/badge/arXiv-2107.05399-b31b1b.svg)]()
[![Survey](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<!-- [![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

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
![Overview](/timeline.png)
|Year|Publication|Title|Abbreviation|Objective|Platform|Network|
|---|---|---|---|---|---|---|
|2015|[ICIP](https://ieeexplore.ieee.org/abstract/document/7351024)|Deepfocal: A method for direct focal length estimation|DeepFocal|Intrinsics|Caffe|AlexNet|
|2015|[ICCV](https://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html)|Posenet: A convolutional network for real-time 6-dof camera relocalization|PoseNet|Extrinsics|Caffe|GoogLeNet|
|2016|[BMVC](https://arxiv.org/abs/1604.02129)|Horizon lines in the wild|DeepHorizon|Extrinsics|Caffe|GoogLeNet|
|2016|[CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhai_Detecting_Vanishing_Points_CVPR_2016_paper.html)|Detecting vanishing points using global image context in a non-manhattan world|DeepVP|Extrinsics|Caffe|AlexNet|
|2016|[ACCV](https://link.springer.com/chapter/10.1007/978-3-319-54187-7_3)|Radial lens distortion correction using convolutional neural networks trained with synthesized images|Rong et al.|Distortion coefficients|Caffe|AlexNet|
|2016|[RSSW](https://arxiv.org/abs/1606.03798)|Deep image homography estimation|DHN|Projection matrixs|Caffe|VGG|
|2017|[CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.html)|Clkn: Cascaded lucas-kanade networks for image alignment|CLKN|Projection matrixs|Torch|CNN + Lucas-Kanade layer|
|2017|[ICCVW](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Nowruzi_Homography_Estimation_From_ICCV_2017_paper.html)|Homography estimation from image pairs with hierarchical convolutional networks|HierarchicalNet|Projection matrixs|TensorFlow|VGG|
|2018|[CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Hold-Geoffroy_A_Perceptual_Measure_CVPR_2018_paper.html)|A perceptual measure for deep single image camera calibration|Hold-Geoffroy et al.|Intrinsics + Extrinsics| |DenseNet|
|2018|[CVMP](https://dl.acm.org/doi/abs/10.1145/3278471.3278479)|DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras|DeepCalib|Intrinsics + Distortion coefficients|TensorFlow|Inception-V3|
|2018|[ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiaoqing_Yin_FishEyeRecNet_A_Multi-Context_ECCV_2018_paper.html)|Fisheyerecnet: A multi-context collaborative deep network for fisheye image rectification|FishEyeRecNet|Distortion coefficients|Caffe|VGG|
|2018|[ICPR](https://ieeexplore.ieee.org/abstract/document/8545218)|Radial lens distortion correction by adding a weight layer with inverted foveal models to convolutional neural networks|Shi et al.|Distortion coefficients|PyTorch|ResNet|
|2018|[ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.html)|Deep fundamental matrix estimation|DeepFM|Projection matrixs|PyTorch|ResNet|
|2018|[ECCVW](https://openaccess.thecvf.com/content_eccv_2018_workshops/w16/html/Poursaeed_Deep_Fundamental_Matrix_Estimation_without_Correspondences_ECCVW_2018_paper.html)|Deep fundamental matrix estimation without correspondences|Poursaeed et al.|Projection matrixs| |CNNs|
|2018|[RAL](https://ieeexplore.ieee.org/abstract/document/8302515)|Unsupervised deep homography: A fast and robust homography estimation model|UDHN|Projection matrixs|TensorFlow|VGG|
|2018|[ACCV](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_36)|Rethinking planar homography estimation using perspective fields|PFNet|Projection matrixs|TensorFlow|FCN|
|2019|[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Lopez_Deep_Single_Image_Camera_Calibration_With_Radial_Distortion_CVPR_2019_paper.html)|Deep single image camera calibration with radial distortion|Lopez et al.|Intrinsics + Extrinsics + Distortion coefficients|PyTorch|DenseNet|
|2019|[ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Xian_UprightNet_Geometry-Aware_Camera_Orientation_Estimation_From_Single_Images_ICCV_2019_paper.html)|UprightNet: geometry-aware camera orientation estimation from single images|UprightNet|Extrinsics|PyTorch|U-Net|
|2019|[IROS](https://ieeexplore.ieee.org/abstract/document/8967912)|Degeneracy in self-calibration revisited and a deep learning solution for uncalibrated slam|Zhuang et al.|Intrinsics + Distortion coefficients|PyTorch|ResNet|
|2019|[PRL](https://www.sciencedirect.com/science/article/abs/pii/S0141938221001062)|Self-Supervised deep homography estimation with invertibility constraints|SSR-Net|Projection matrixs|PyTorch|ResNet|
|2019|[ICCVW](https://openaccess.thecvf.com/content_ICCVW_2019/html/GMDL/Abbas_A_Geometric_Approach_to_Obtain_a_Birds_Eye_View_From_ICCVW_2019_paper.html)|A geometric approach to obtain a bird's eye view from an image|Abbas et al.|Projection matrixs|TensorFlow|CNNs|
|2019|[TCSVT](https://ieeexplore.ieee.org/abstract/document/8636975)|DR-GAN: Automatic radial distortion rectification using conditional GAN in real-time|DR-GAN|Undistortion|TensorFlow|GANs|
|2019|[TCSVT](https://ieeexplore.ieee.org/abstract/document/8926530)|Distortion rectification from static to dynamic: A distortion sequence construction perspective|STD|Undistortion|TensorFlow|GANs|
|2019|[VR](https://ieeexplore.ieee.org/abstract/document/8798326)|Deep360Up: A deep learning-based approach for automatic VR image upright adjustment|Deep360Up|Extrinsics| |DenseNet|
|2019|[JVCIR](https://www.sciencedirect.com/science/article/abs/pii/S104732031930313X)|Unsupervised fisheye image correction through bidirectional loss with geometric prior|UnFishCor|Distortion coefficients|TensorFlow|VGG|
|2019|[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Blind_Geometric_Distortion_Correction_on_Images_Through_Deep_Learning_CVPR_2019_paper.html)|Blind geometric distortion correction on images through deep learning|BlindCor|Undistortion|PyTorch|U-Net|
|2019|[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhuang_Learning_Structure-And-Motion-Aware_Rolling_Shutter_Correction_CVPR_2019_paper.html)|Learning structure-and-motion-aware rolling shutter correction|RSC-Net|Undistortion|PyTorch|ResNet|
|2019|[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Xue_Learning_to_Calibrate_Straight_Lines_for_Fisheye_Image_Rectification_CVPR_2019_paper.html)|Learning to calibrate straight lines for fisheye image rectification|Xue et al.|Distortion coefficients|PyTorch|ResNet|
|2019|[ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhao_Learning_Perspective_Undistortion_of_Portraits_ICCV_2019_paper.html)|Learning perspective undistortion of portraits|Zhao et al.|Intrinsics + Undistortion||VGG + U-Net|
|2020|[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.html)|End-to-end camera calibration for broadcast videos|Sha et al.|Projection matrixs|TensorFlow|Siamese-Net + U-Net|
|2020|[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_32)|Neural geometric parser for single image camera calibration|Lee et al.|Intrinsics + Extrinsics| |PointNet + CNNs|
|2020|[ICRA](https://ieeexplore.ieee.org/abstract/document/9197378)|Learning camera miscalibration detection|MisCaliDet|Average pixel position difference| |CNNs|
|2020|[WACV](https://openaccess.thecvf.com/content_WACV_2020/html/Zhang_DeepPTZ_Deep_Self-Calibration_for_PTZ_Cameras_WACV_2020_paper.html)|DeepPTZ: deep self-calibration for PTZ cameras|DeepPTZ|Intrinsics + Extrinsics + Distortion coefficients|PyTorch|Inception-V3|
|2020|[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Deep_Homography_Estimation_for_Dynamic_Scenes_CVPR_2020_paper.html)|Deep homography estimation for dynamic scenes|MHN|Projection matrixs|TensorFlow|VGG|
|2020|[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58604-1_35)|360∘ camera alignment via segmentation|Davidson et al.|Extrinsics| |FCN|
|2020|[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_38)|Content-aware unsupervised deep homography estimation|CA-UDHN|Projection matrixs|PyTorch|FCN + ResNet|
|2020|[IROS](https://ieeexplore.ieee.org/abstract/document/9341229)|Deep keypoint-based camera pose estimation with geometric constraints|DeepFEPE|Extrinsics|PyTorch|VGG + PointNet|
|2020|[TIP](https://ieeexplore.ieee.org/abstract/document/8962122)|Model-free distortion rectification framework bridged by distortion distribution map|DDM|Undistortion|Tensorflow|GANs|
|2020|[TIP](https://ieeexplore.ieee.org/abstract/document/9184235)|Deep face rectification for 360° dual-fisheye cameras|Li et al.|Undistortion| |CNNs|
|2020|[ICPR](https://ieeexplore.ieee.org/abstract/document/9412305)|Position-aware and symmetry enhanced GAN for radial distortion correction|PSE-GAN|Undistortion| |GANs|
|2020|[ICIP](https://ieeexplore.ieee.org/abstract/document/9191107)|A simple yet effective pipeline for radial distortion correction|RDC-Net|Undistortion|PyTorch|ResNet|
|2020|[ICASSP](https://ieeexplore.ieee.org/abstract/document/9054191)|Self-supervised deep learning for fisheye image rectification|FE-GAN|Undistortion|PyTorch|GANs|
|2020|[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_RDCFace_Radial_Distortion_Correction_for_Face_Recognition_CVPR_2020_paper.html)|RDCFace: radial distortion correction for face recognition|RDCFace|Undistortion| |ResNet|
|2020|[arXiv](https://arxiv.org/abs/2003.11386)|Fisheye distortion rectification from deep straight lines|LaRecNet|Distortion coefficients|PyTorch|ResNet|
|2020|[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Baradad_Height_and_Uprightness_Invariance_for_3D_Prediction_From_a_Single_CVPR_2020_paper.html)|Height and uprightness invariance for 3d prediction from a single view|Baradad et al.|Intrinsics + Extrinsics|PyTorch|CNNs|
|2020|[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_What_Does_Plate_Glass_Reveal_About_Camera_Calibration_CVPR_2020_paper.html)|What does plate glass reveal about camera calibration?|Zheng et al.|Intrinsics + Extrinsics| |CNNs|
|2020|[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_19)|Single view metrology in the wild|Zhu et al.|Intrinsics + Extrinsics|PyTorch|CNNs + PointNet|
|2021|[TCI](https://ieeexplore.ieee.org/abstract/document/9495157)|Online training of stereo self-calibration using monocular depth estimation|StereoCaliNet|Extrinsics|PyTorch|U-Net|
|2021|[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Lee_CTRL-C_Camera_Calibration_TRansformer_With_Line-Classification_ICCV_2021_paper.html?ref=https://githubhelp.com)|CTRL-C: Camera calibration TRansformer with Line-Classification|CTRL-C|Intrinsics + Extrinsics|PyTorch|Transformer|
|2021|[ICCVW](https://openaccess.thecvf.com/content/ICCV2021W/PBDL/html/Wakai_Deep_Single_Fisheye_Image_Camera_Calibration_for_Over_180-Degree_Projection_ICCVW_2021_paper.html)|Deep single fisheye image camera calibration for over 180-degree projection of field of view|Wakai et al.|Intrinsics + Extrinsics| |DenseNet|
|2021|[arXiv](https://arxiv.org/abs/2111.12927)|Rethinking generic camera models for deep single image camera calibration to recover rotation and fisheye distortion|GenCaliNet|Intrinsics + Extrinsics + Distortion coefficients| |DenseNet|
|2021|[TIP](https://ieeexplore.ieee.org/abstract/document/9366359)|A deep ordinal distortion estimation approach for distortion rectification|OrdianlDistortion|Distortion coefficients|TensorFlow|CNNs|
|2021|[TCSVT](https://ieeexplore.ieee.org/abstract/document/9567670)|Revisiting radial distortion rectification in polar-coordinates: A new and efficient learning perspective|PolarRecNet|Undistortion|PyTorch|VGG + U-Net|
|2021|[PRL](https://www.sciencedirect.com/science/article/abs/pii/S0167865521003299)|DQN-based gradual fisheye image rectification|DQN-RecNet|Undistortion|PyTorch|VGG|
|2021|[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Tan_Practical_Wide-Angle_Portraits_Correction_With_Deep_Structured_Models_CVPR_2021_paper.html)|Practical wide-angle portraits correction with deep structured models|Tan et al.|Undistortion|PyTorch|U-Net|
|2021|[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Progressively_Complementary_Network_for_Fisheye_Image_Rectification_Using_Appearance_Flow_CVPR_2021_paper.html)|Progressively complementary network for fisheye image rectification using appearance flow|PCN|Undistortion|PyTorch|U-Net|
|2021|[arXiv](https://arxiv.org/abs/2011.14611)|SIR: Self-supervised image rectification via seeing the same scene from multiple different lenses|SIR|Undistortion|PyTorch|ResNet|
|2021|[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Liao_Multi-Level_Curriculum_for_Training_a_Distortion-Aware_Barrel_Distortion_Rectification_Model_ICCV_2021_paper.html)|Multi-level curriculum for training a distortion-aware barrel distortion rectification model|DaRecNet|Undistortion|TensorFlow|U-Net|
|2021|[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Deep_Lucas-Kanade_Homography_for_Multimodal_Image_Alignment_CVPR_2021_paper.html)|Deep Lucas-Kanade homography for multimodal image alignment|DLKFM|Projection matrixs|TensorFlow|Siamese-Net|
|2021|[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Shao_LocalTrans_A_Multiscale_Local_Transformer_Network_for_Cross-Resolution_Homography_Estimation_ICCV_2021_paper.html)|LocalTrans: A multiscale local transformer network for cross-resolution homography estimation|LocalTrans|Projection matrixs|PyTorch|Transformer|
|2021|[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Ye_Motion_Basis_Learning_for_Unsupervised_Deep_Homography_Estimation_With_Subspace_ICCV_2021_paper.html)|Motion basis learning for unsupervised deep homography estimation with subspace projection|BasesHomo|Projection matrixs|PyTorch|ResNet|
|2021|[ICIP](https://ieeexplore.ieee.org/abstract/document/9506264)|Fast and accurate homography estimation using extendable compression network|ShuffleHomoNet|Projection matrixs|TensorFlow|ShuffleNet|
|2021|[TCSVT](https://arxiv.org/abs/2107.02524)|Depth-aware multi-grid deep homography estimation with contextual correlation|DAMG-Homo|Projection matrixs|TensorFlow|CNNs|
|2021|[BMVC](https://www.bmvc2021-virtualconference.com/assets/papers/1364.pdf)|A simple approach to image tilt correction with self-attention MobileNet for smartphones|SA-MobileNet|Extrinsics|TensorFlow|MobileNet|
|2021|[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Kocabas_SPEC_Seeing_People_in_the_Wild_With_an_Estimated_Camera_ICCV_2021_paper.html)|SPEC: Seeing people in the wild with an estimated camera|SPEC|Intrinsics + Extrinsics|PyTorch|ResNet|
|2021|[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Wide-Baseline_Relative_Camera_Pose_Estimation_With_Directional_Learning_CVPR_2021_paper.pdf)|Wide-Baseline Relative Camera Pose Estimation with Directional Learning|DirectionNet|Extrinsics|TensorFlow|U-Net|
|2022|[CVPR](https://arxiv.org/abs/2203.08586)|Deep vanishing point detection: Geometric priors make dataset variations vanish|DVPD|Extrinsics|PyTorch|CNNs|
|2022|[ICRA](https://arxiv.org/abs/2112.03325)|Self-supervised camera self-calibration from video|Fang et al.|Intrinsics + Extrinsics|PyTorch|CNNs|
|2022|[ICASSP](https://ieeexplore.ieee.org/abstract/document/9746819)|Camera calibration through camera projection loss|CPL|Intrinsics + Extrinsics|TensorFlow|Inception-V3|
|2022|[CVPR](https://arxiv.org/abs/2203.15982)|Iterative Deep Homography Estimation|IHN|Projection matrixs|PyTorch|Siamese-Net|
|2022|[CVPR](https://arxiv.org/abs/2205.03821)|Unsupervised Homography Estimation with Coplanarity-Aware GAN|HomoGAN|Projection matrixs|PyTorch|GANs|
|2022|[CVPR](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Semi-Supervised_Wide-Angle_Portraits_Correction_by_Multi-Scale_Transformer_CVPR_2022_paper.pdf)|Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer|SS-WPC|Undistortion|PyTorch|Transformer|
|2022|[PAMI](https://ieeexplore.ieee.org/abstract/document/9771389)|Content-Aware Unsupervised Deep Homography Estimation and Beyond|Liu et al.|Projection matrixs|PyTorch|ResNet|

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
@article{Kang,
  title={Deep Learning for Camera Calibration: A Survey},
  author={Kang Liao, Chunyu Lin, Yunchao Wei, Yao Zhao},
  journal={},
  year={2022}
}
```
## 📋Paper
[Official Version]()

[arXiv Version]()


## 📭Contact

```
kang_liao@bjtu.edu.cn
```
