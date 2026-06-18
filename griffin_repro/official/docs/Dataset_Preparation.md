# 📊 Dataset Preparation

Griffin is split into three subsets based on UAV cruising altitude:
- **Griffin-Random**: 20–60 meters (104 scenes)
- **Griffin-25m**: 25 ± 2 meters (47 scenes)
- **Griffin-40m**: 40 ± 2 meters (54 scenes)
- **Griffin-55m**: 55 ± 2 meters (50 scenes)

Each of the 255 scenes lasts ~15 seconds (~150 frames), totaling over 37.7k samples, 339.3k images, and 914.8k 3D annotations. The dataset supports both KITTI (ego-centric) and NuScenes (global reference) formats.

## 📥 Download Dataset

Get Griffin from [Baidu Netdisk](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm) or [Hugging Face](https://huggingface.co/datasets/wjh-svm/Griffin)

## 📁 Dataset Organization

After downloading, unzip them and organize your dataset directory as shown below. The example uses `griffin_50scenes_25m`, but all subsets follow the same structure:
### KITTI-Style Raw Data Structure

```
  └── datasets
      ├── griffin_50scenes_25m
      │   └── griffin-release
      │       ├── vehicle-side
      │       │   ├── calib
      │       │   ├── camera
      │       │   │   ├── back
      │       │   │   ├── front
      │       │   │   ├── instance_back
      │       │   │   ├── instance_front
      │       │   │   ├── instance_left
      │       │   │   ├── instance_right
      │       │   │   ├── left
      │       │   │   └── right
      │       │   ├── label
      │       │   ├── lidar
      │       │   │   └── lidar_top
      │       │   ├── pose
      │       │   └── scene_infos.json
      │       └── drone-side
      │           ├── calib
      │           ├── camera
      │           │   ├── back
      │           │   ├── bottom
      │           │   ├── front
      │           │   ├── instance_back
      │           │   ├── instance_bottom
      │           │   ├── instance_front
      │           │   ├── instance_left
      │           │   ├── instance_right
      │           │   ├── left
      │           │   └── right
      │           ├── label
      │           ├── pose
      │           └── scene_infos.json
      ├── griffin_50scenes_40m
      ├── griffin_50scenes_55m
      └── griffin_100scenes_random
```

## 🔄 Convert to NuScenes Format

Transform the KITTI-style raw data to NuScenes format for cooperative perception:

### Quick Conversion Commands

```bash
# Convert all Griffin subsets to NuScenes format
bash tools/griffin_converter.sh griffin_50scenes_25m
bash tools/griffin_converter.sh griffin_50scenes_40m 
bash tools/griffin_converter.sh griffin_50scenes_55m 
bash tools/griffin_converter.sh griffin_100scenes_random
```

> ⏱️ **Processing time**: ~5-15 minutes per subset depending on system performance

### Conversion Process

The converter script performs these operations:
1. **Extracts metadata** from `scene_infos.json` files
2. **Creates relational database** structure with JSON files
3. **Generates four cooperative perspectives**: vehicle-side, drone-side, early-fusion, cooperative
4. **Preserves all annotation** and calibration data

### NuScenes Format Structure

After conversion, you'll have this organized structure:
```
  .
  ├── data
  │   ├── infos
  │   │   └── griffin_50scenes_25m
  │   │       ├── cooperative
  │   │       ├── drone-side
  │   │       └── vehicle-side
  │   └── split_datas
  │       └── griffin_50scenes_25m.json
  └── datasets
      └── griffin_50scenes_25m
          ├── griffin-nuscenes
          │   ├── vehicle-side
          │   │   ├── v1.0-trainval
          │   │   │   ├── calibrated_sensor.json
          │   │   │   ├── ego_pose.json
          │   │   │   ├── sample.json
          │   │   │   ├── sample_annotation.json
          │   │   │   ├── sample_data.json
          │   │   │   └── ... (other metadata)
          │   │   ├── maps
          │   │   └── samples
          │   │       ├── CAM_BACK
          │   │       ├── CAM_FRONT
          │   │       ├── CAM_LEFT
          │   │       └── CAM_RIGHT
          │   ├── drone-side
          │   │   ├── v1.0-trainval
          │   │   ├── maps
          │   │   └── samples
          │   │       ├── CAM_BACK
          │   │       ├── CAM_BOTTOM
          │   │       ├── CAM_FRONT
          │   │       ├── CAM_LEFT
          │   │       └── CAM_RIGHT
          │   ├── early-fusion
          │   │   ├── v1.0-trainval
          │   │   ├── maps
          │   │   └── samples
          │   │       ├── CAM_BACK
          │   │       ├── CAM_BACK_AIR
          │   │       ├── CAM_BOTTOM_AIR
          │   │       ├── CAM_FRONT
          │   │       ├── CAM_FRONT_AIR
          │   │       └── ... (all perspectives)
          │   └── cooperative
          │       ├── v1.0-trainval
          │       ├── maps
          │       └── samples
          └── griffin-release
              ├── vehicle-side
              └── drone-side
```

## 🔍 Data Format Details

The KITTI-style structure, provided in the griffin-release directory, offers an intuitive file-based organization. Data is segregated into vehicle-side and drone-side folders, each containing subdirectories for raw sensor data and annotations. These include:
- camera: Contains RGB images from all camera perspectives (e.g., front, left, right, back, and bottom for the drone). It also includes corresponding pixel-level instance segmentation maps in instance_*/ subfolders.
- lidar: Stores LiDAR point cloud data (.bin files), available only for the ground vehicle.
- calib: Provides sensor calibration parameters, including intrinsics and extrinsics, in .json files.
- label: Contains frame-by-frame 3D bounding box annotations.
- pose: Includes the ego-pose for each agent at every timestamp.

The NuScenes-style structure, located in the griffin-nuscenes directory, organizes the data in a relational database format. A set of .json files (e.g., sample.json, sample_data.json, sample_annotation.json) defines the relationships between scenes, data samples, sensor captures, and annotations. This format is highly flexible and supports complex, database-style queries. A key feature of our release is the provision of four pre-structured perspectives to streamline experimentation:

**vehicle-side and drone-side**: Contain data exclusively for the ground vehicle and drone, respectively, ideal for single-agent baseline training and evaluation.

**early-fusion and cooperative**: Provide a unified data structure where samples from both the vehicle and drone are aggregated under a single frame of reference. This organization simplifies the setup for cooperative perception experiments by pre-aligning multi-agent sensor data (e.g., CAM_FRONT, CAM_FRONT_AIR), allowing researchers to focus directly on fusion methodologies.

## 🎯 Next Steps

With your dataset ready, you can now:

1. **🏋️ Start Training**: [Training and Evaluation Guide](Training_and_Evaluation.md)
2. **👁️ Visualize Data**: [Visualization Guide](Visualization.md)
3. **📊 Explore Results**: Check out our [detailed results](detailed_results.csv)

Ready to dive into aerial-ground cooperative perception! 🚁🚗