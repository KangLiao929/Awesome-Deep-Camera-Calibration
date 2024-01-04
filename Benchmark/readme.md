# Benchmark for Learning-based Camera Calibration and Beyond

## :circus_tent: Standard Pinhole Model 

### Brief Description

We collected 300 high-resolution images on the Internet [1], captured by popular digital cameras such as Canon, Fujifilm, Nikon, Olympus, Sigma, Sony, etc. For each image, we provide the specific focal length of its lens. We have included a diverse range of subjects, including landscapes, portraits, wildlife, architecture, etc. The range of focal length is from 4.5mm to 600mm, allowing users to explore the different effects that different lenses can have on an image.

### Directory/Data Structure and Parsing

## :circus_tent: Distortion Camera Model 

### Brief Description
We created a comprehensive dataset for the distortion camera model, with a focus on wide-angle cameras. The dataset is comprised of three subcategories. The first is a synthetic dataset, which was generated using the widely-used 4<sup>th</sup> order polynomial model. It contains both circular and rectangular structures, with 1,000 distortion-rectification image pairs. The second subcategory consists of data captured under real-world settings, derived from the raw calibration data for around 40 types of wide-angle cameras. For each calibration data, the intrinsics, extrinsics, and distortion coefficients are available. Finally, we exploit a car equipped with different cameras to capture video sequences. The scenes cover both indoor and outdoor environments, including daytime and nighttime footage.

### Directory/Data Structure and Parsing


## :circus_tent: Cross-View Model

### Brief Description

We selected 500 testing samples at random from each of the four representative datasets to create a dataset for the cross-view model. It covers a range of scenarios: MS-COCO provides natural synthetic data, GoogleEarch contains aerial synthetic data, and GoogleMap offers multi-modal synthetic data. Parallax is not a factor in these three datasets, while CAHomo provides real-world data with non-planar scenes. To standardize the dataset, we converted all images to a unified format and recorded the matched points between two views. In MS-COCO, GoogleEarch, and GoogleMap, we used four vertices of the images as the matched points. In CAHomo, we identified six matched key points within the same plane.

### Directory/Data Structure and Parsing

## :circus_tent: Cross-Sensor Model

### Brief Description

We collected RGB and point cloud data from Apollo, DAIR-V2X, KITTI, KUCL, NuScenes, and ONCE. Around 300 data pairs with calibration parameters are included in each category. The datasets are captured in different countries to provide enough variety. Each dataset has a different sensor setup, obtaining camera-LiDAR data with varying image resolution, LiDAR scan pattern, and camera-LiDAR relative location. The image resolution ranges from 2448x2048 to 1242x375, while the LiDAR sensors are from Velodyne and Hesai, with 16, 32, 40, 64, and 128 beams. They include not only normal surrounding multi-view images but also small baseline multi-view data. Additionally, we also added random disturbance of around 20 degrees rotation and 1.5 meters translation based on classical settings to simulate vibration and collision.

### Directory/Data Structure and Parsing



## Reference
```bash
[1] https://www.dpreview.com/sample-galleries
```


