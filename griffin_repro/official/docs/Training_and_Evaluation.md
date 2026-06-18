# 🏋️ Training and Evaluation Guide

Griffin supports multiple paradigms for aerial-ground cooperative perception. This guide covers everything from single-agent baselines to advanced cooperative models.

## 🎯 Overview

Griffin provides several approaches to explore cooperative perception:

| Approach                | Agents                        |
| ----------------------- | ----------------------------- |
| **Vehicle-Side**        | Ground vehicle only           |
| **Drone-Side**          | UAV/drone only                |
| **Early-Fusion**        | Both (sensor-level)           |
| **Intermediate-Fusion** | Both (instance-feature-level) |
| **Late-Fusion**         | Both (result-level)           |

### Prerequisites
- ✅ Griffin installation completed ([Installation Guide](Installation.md))
- ✅ Dataset downloaded and converted ([Dataset Preparation](Dataset_Preparation.md))
- ✅ Pre-trained [BEVFormer](https://github.com/fundamentalvision/BEVFormer) checkpoint downloaded (see below)

---

## 📥 Download Required Checkpoints

Before starting training, you need to download the BEVFormer base checkpoint that Griffin builds upon:

### BEVFormer Base Checkpoint

**Required for all training**: Download the BEVFormer tiny checkpoint and organize as follows:

```bash
# Create checkpoint directory
mkdir -p ckpts/

# Download BEVFormer checkpoint into 'ckpts' folder (choose one option)
```

**Option 1**: Griffin Release (includes all pre-trained models) from [Baidu Netdisk](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm) or [Hugging Face](https://huggingface.co/datasets/wjh-svm/Griffin)

**Option 2**: [Official BEVFormer Release](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth)

### Verify Checkpoint

Verify the checkpoint exists and has correct MD5

```bash
md5sum ckpts/bevformer_tiny_epoch_24.pth
# Expected: 859353d9e740a9870b233efb1b0d27d4
```

> 📌 **Note**: This checkpoint is essential for all Griffin training workflows. It provides the base BEVFormer architecture that Griffin extends for cooperative perception.

---

## 🚀 Training Workflows

For advanced cooperative perception research, follow this sequential process:

### **Step 1: 🚗 Vehicle-Side Model Training**

Train the ground vehicle perception model as the foundation:

```bash
CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_VEHICLE NUM_GPUS
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh projects/configs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls.py 4
```

#### **Step 2: 🚁 Drone-Side Model Training & Query Extraction**

Train the drone model and save cooperative features:

```bash
# 2a. Train drone-side model
CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_DRONE NUM_GPUS
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls.py 4

# 2b. Extract track queries for training set
CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_eval.sh CONFIG_FILE_DRONE_EVAL_TRAIN NUM_GPUS
# CUDA_VISIBLE_DEVICES=0 ./tools/dist_eval.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls/latest.pth 1

# 2c. Extract track queries for validation set
CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_eval.sh CONFIG_FILE_DRONE_EVAL NUM_GPUS
# CUDA_VISIBLE_DEVICES=0 ./tools/dist_eval.sh projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls/latest.pth 1
```

#### **Step 3: 🤝 Cooperative Model Training**

Train the cooperative fusion model using extracted drone queries:

```bash
CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_COOP NUM_GPUS
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py 4
```

## 📊 Evaluation

Griffin provides pre-trained checkpoints for quick evaluation and comparison. Download from [Baidu Netdisk](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm) or [Hugging Face](https://huggingface.co/datasets/wjh-svm/Griffin).

### 📥 Pre-trained Model Structure

```
ckpts/
├── bevformer_tiny_epoch_24.pth (md5sum: 859353d9e740a9870b233efb1b0d27d4)
├── griffin_100scenes_random
│   ├── cooperative
│   │   └── instance_fusion
│   │       └── iter_36072.pth  (md5sum: 91a144ab62e63e7b214eadce4d2aeaa8)
│   ├── drone-side
│   │   └── iter_36072.pth      (md5sum: 0267770d66d2e66568728a3c67b0e48e)
│   ├── early-fusion
│   │   └── iter_36072.pth      (md5sum: 319aaca5cf809b28507336a5ed61f12d)
│   └── vehicle-side
│       └── iter_36072.pth      (md5sum: 44a96afcaf31b18b0cd21fe37e488f37)
├── griffin_50scenes_25m
│   ├── cooperative
│   │   └── instance_fusion
│   │       └── iter_33024.pth  (md5sum: 7e1448188b6e99ca6303575c3466b97f)
│   ├── drone-side
│   │   └── iter_33024.pth      (md5sum: 41734b8d764d1213935a231a3419f655)
│   ├── early-fusion
│   │   └── iter_33024.pth      (md5sum: ccf39651d3e4381ec06e4d0709821949)
│   └── vehicle-side
│       └── iter_33024.pth      (md5sum: 1201f5692e390a75e5b5ab0efa0cae19)
├── griffin_50scenes_40m
│   ├── cooperative
│   │   └── instance_fusion
│   │       └── iter_38784.pth  (md5sum: c4f1e25c8cac1e4bd024d0677f2b86af)
│   ├── drone-side
│   │   └── iter_38784.pth      (md5sum: 8e078f4d2d09c099e0915293c51273bd)
│   ├── early-fusion
│   │   └── iter_38784.pth      (md5sum: 8711c164071a27d8864b942940ce530b)
│   └── vehicle-side
│       └── iter_38784.pth      (md5sum: 3089d797c511781dcff0e6d415a25d8b)
└── griffin_50scenes_55m
    ├── cooperative
    │   └── instance_fusion
    │       └── iter_35760.pth  (md5sum: ebd68b971f814705c0c708b20e62b37e)
    ├── drone-side
    │   └── iter_35760.pth      (md5sum: 5375f13024116131c281fbcd9c80fd00)
    ├── early-fusion
    │   └── iter_35760.pth      (md5sum: 6a19c8c3758cc06c0fa23f26d020ee06)
    └── vehicle-side
        └── iter_35760.pth      (md5sum: 3a9bf390edd4855f7d9ab1d3fcaea3e0)
```

### 🧪 Evaluation Commands

**🚗 Vehicle-Side Model**:
```bash
./tools/dist_eval.sh CONFIG_FILE_VEHICLE CHECKPOINT_VEHICLE 1
```

**🚁 Drone-Side Model**:
```bash
./tools/dist_eval.sh CONFIG_FILE_DRONE CHECKPOINT_DRONE 1
```

**🤝 Cooperative Model**:
```bash
./tools/dist_eval.sh CONFIG_FILE_COOP CHECKPOINT_COOP 1
```

---

## 🛡️ Robustness Testing

Griffin includes comprehensive robustness evaluation to test cooperative models under real-world conditions:

### Available Test Conditions

| Test Type                   | Configs Available            | Purpose                   |
| --------------------------- | ---------------------------- | ------------------------- |
| **📦 Packet Loss**           | 10%, 20%, 30%, 40%, 50%      | Communication reliability |
| **⏱️ Communication Latency** | 100ms, 200ms, 300ms, 400ms   | Network delay impact      |
| **📍 Localization Error**    | 0.5m, 1.0m, 1.5m, 2.0m, 2.5m | GPS/positioning noise     |
| **🧭 Orientation Error**     | 1°, 2°, 3°, 4°, 5°           | Heading/compass noise     |

### Robustness Config Structure

```
📁 projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/
├── 📁 drop_noised/                   📦 Packet loss testing
│   ├── 📄 *_drop10.py                10% packet loss
│   ├── 📄 *_drop20.py                20% packet loss  
│   ├── 📄 *_drop30.py                30% packet loss
│   ├── 📄 *_drop40.py                40% packet loss
│   └── 📄 *_drop50.py                50% packet loss
├── 📁 latency/                       ⏱️ Communication delay testing
│   ├── 📄 *_100latency.py            100ms latency
│   ├── 📄 *_200latency.py            200ms latency
│   ├── 📄 *_300latency.py            300ms latency
│   └── 📄 *_400latency.py            400ms latency
├── 📁 loc_noised/                    📍 Localization noise testing
│   ├── 📄 *_loc05.py                 0.5m positioning error
│   ├── 📄 *_loc10.py                 1.0m positioning error
│   ├── 📄 *_loc15.py                 1.5m positioning error
│   ├── 📄 *_loc20.py                 2.0m positioning error
│   └── 📄 *_loc25.py                 2.5m positioning error
├── 📁 orien_noised/                  🧭 Orientation noise testing
│   ├── 📄 *_orien1.py                1° orientation error
│   ├── 📄 *_orien2.py                2° orientation error
│   ├── 📄 *_orien3.py                3° orientation error
│   ├── 📄 *_orien4.py                4° orientation error
│   └── 📄 *_orien5.py                5° orientation error
└── 📄 tiny_track_r50_stream_bs8_48epoch_3cls.py  📊 Baseline (no noise)
```

### Running Robustness Tests

**Example: Test 20% packet loss**:
```bash
./tools/dist_eval.sh projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/drop_noised/tiny_track_r50_stream_bs8_48epoch_3cls_drop20.py ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth 1
```

---

## 💡 Tips for Success

### 🚀 Training Tips
1. **Monitor GPU memory**: Reduce batch size if OOM errors occur
2. **Use tensorboard**: Track training progress with `tensorboard --logdir work_dirs/`
3. **Save checkpoints**: Models auto-save every few epochs
4. **Validate early**: Check performance on small subset first

### 🔧 Debugging Common Issues

| Issue                | Cause                | Solution                                                                           |
| -------------------- | -------------------- | ---------------------------------------------------------------------------------- |
| **OOM Error**        | Batch size too large | Reduce `batch_size` and `lr` in config                                             |
| **Slow training**    | I/O bottleneck       | Use SSD storage, increase `num_workers`                                            |
| **Poor performance** | Wrong learning rate  | Check total batch size (num_gpus × batch_size) and adjust lr almost proportionally |
| **Config errors**    | Path mismatch        | Check dataset paths in config files                                                |

---

## 🔗 Next Steps

After training and evaluation:

1. **👁️ Visualize Results**: Use our [Visualization Guide](Visualization.md)
2. **📊 Analyze Performance**: Check [detailed_results.csv](detailed_results.csv)

Happy training! 🚁🚗
