import nuscenes

dataset_nusc_paths = [
    "datasets/griffin_50scenes_25m/griffin-nuscenes/early-fusion",
    "datasets/griffin_50scenes_40m/griffin-nuscenes/early-fusion",
    "datasets/griffin_50scenes_55m/griffin-nuscenes/early-fusion",
    "datasets/griffin_100scenes_random/griffin-nuscenes/early-fusion",
]

scene_count = 0
frame_count = 0
instance_count = 0
bbox_count = 0

for dataset_nusc_path in dataset_nusc_paths:
    print("-" * 100)
    print(f"Processing {dataset_nusc_path}")
    nusc = nuscenes.NuScenes(
        version='v1.0-trainval', dataroot=dataset_nusc_path, verbose=True
    )
    scene_count += len(nusc.scene)
    frame_count += len(nusc.sample)
    instance_count += len(nusc.instance)
    bbox_count += len(nusc.sample_annotation)

print("-" * 100)
print(f"scene_count: {scene_count}")
print(f"frame_count: {frame_count}")
print(f"image_count: {frame_count * 9}")
print(f"instance_count: {instance_count}")
print(f"bbox_count: {bbox_count}")
