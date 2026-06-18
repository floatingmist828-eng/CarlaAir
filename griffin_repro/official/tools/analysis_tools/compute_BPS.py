import pickle
import torch
import os
import random
from tqdm import tqdm
import numpy as np

v2xvit_bev_path = "data/infos/griffin_50scenes_25m/drone-side/track_bev"
where2comm_bev_path = "data/infos/griffin_50scenes_25m/drone-side/track_bev_where2comm"
univ2x_instance_path = "data/infos/griffin_50scenes_25m/drone-side/track_query"
cooptrack_instance_path = (
    "data/infos/cooptrack_griffin_50scenes_25m/drone-side_load_det/track_query"
)
late_path = "projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval/results-07031922.pkl"

data_frequency = 10
sample_frame_num = -1

# Random sample frames
with open(late_path, 'rb') as f:
    print(f"Loading Late Fusion Results")
    data = pickle.load(f)
sample_list = [
    (idx, result['token']) for idx, result in enumerate(data['bbox_results'])
]

if sample_frame_num > 0:
    print(f"Sampling {sample_frame_num} out of {len(sample_list)} frames")
    sample_list = random.sample(sample_list, sample_frame_num)
else:
    sample_frame_num = len(sample_list)


# Late Fusion BPS
bt = 0
result_num_sum = 0
for idx, sample in tqdm(sample_list, desc="Calculating Late Fusion BPS"):
    result = data['bbox_results'][idx]
    result_num = len(result['labels_3d'])
    result_num_sum += result_num

    assert (
        result['labels_3d'].dtype == torch.int32
        or result['labels_3d'].dtype == torch.int64
    )
    bt_per_number = 1
    bt += bt_per_number * result_num

    assert result['scores_3d'].dtype == torch.float32
    bt_per_number = 4
    bt += bt_per_number * result_num

    assert result['boxes_3d'].center.dtype == torch.float32
    bt_per_number = 4
    bt += bt_per_number * result_num * 7  # 7: x, y, z, dim_x, dim_y, dim_z, heading

bt_per_second = bt / sample_frame_num * data_frequency
result_num_avg = result_num_sum / sample_frame_num
print(f"Late Fusion BPS: {bt_per_second}, Result Per Frame: {result_num_avg}")


# V2X-ViT BPS
bt = 0
compress_rate = 32
for _, sample in tqdm(sample_list, desc="Calculating BEV Fusion BPS"):
    with open(os.path.join(v2xvit_bev_path, sample + ".pkl"), 'rb') as f:
        data = pickle.load(f)

    for _, value in data.items():
        assert value.dtype == torch.float32
        bt_per_number = 4
        for s in value.shape:
            bt_per_number *= s
        bt += bt_per_number / 32
bt_per_second = bt / sample_frame_num * data_frequency
print(f"BEV Fusion BPS: {bt_per_second}")


# Where2Comm BPS
def init_gaussian_filter():
    kernel_size = 5
    c_sigma = 1.0

    gaussian_filter = torch.nn.Conv2d(
        1,
        1,
        kernel_size=kernel_size,
        stride=1,
        padding=(kernel_size - 1) // 2,
    )

    center = kernel_size // 2
    x, y = np.mgrid[
        0 - center : kernel_size - center, 0 - center : kernel_size - center
    ]
    gaussian_kernel = (
        1
        / (2 * np.pi * c_sigma)
        * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(c_sigma)))
    )

    gaussian_filter.weight.data = (
        torch.Tensor(gaussian_kernel)
        .to(gaussian_filter.weight.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    gaussian_filter.bias.data.zero_()
    gaussian_filter.requires_grad = False

    return gaussian_filter


# Where2Comm BPS
bt = 0
compress_rate = 4
mask_compress_rate_sum = 0
gaussian_filter = init_gaussian_filter()
for _, sample in tqdm(sample_list, desc="Calculating Where2Comm BPS"):

    with open(os.path.join(where2comm_bev_path, sample + ".pkl"), 'rb') as f:
        data = pickle.load(f)

    inf_bev_mask = data['inf_bev_mask']  # [1,50,50]
    inf_bev_mask = gaussian_filter(inf_bev_mask.unsqueeze(0))
    thre = 0.001
    ones_mask = torch.ones_like(inf_bev_mask).to(inf_bev_mask.device)
    zeros_mask = torch.zeros_like(inf_bev_mask).to(inf_bev_mask.device)
    comm_mask = torch.where(inf_bev_mask > thre, ones_mask, zeros_mask)
    mask_compress_rate = inf_bev_mask.numel() / comm_mask.sum()
    mask_compress_rate_sum += mask_compress_rate

    value = data['inf_bev']  # [2500, 256]
    assert value.dtype == torch.float32
    bt_per_number = 4
    for s in value.shape:
        bt_per_number *= s

    bt += bt_per_number / compress_rate / mask_compress_rate

bt_per_second = bt / sample_frame_num * data_frequency
mask_compress_rate_avg = mask_compress_rate_sum / sample_frame_num
print(f"Where2Comm BPS: {bt_per_second}, Mask Compress Rate: {mask_compress_rate_avg}")


# UniV2X BPS
bt = 0
result_num_sum = 0
inf_keys = [
    'query_feats',
    'obj_idxes',
    'ref_pts',
]  # Remove 'query_embeds', as it can be calculated from ref_pts
for _, sample in tqdm(sample_list, desc="Calculating UniV2X Instance BPS"):
    with open(os.path.join(univ2x_instance_path, sample + ".pkl"), 'rb') as f:
        data = pickle.load(f)

    valid_mask = data.get('obj_idxes') >= 0
    result_num_sum += valid_mask.sum()
    for key in inf_keys:
        value = data.get(key)
        if value is not None:
            value = value[valid_mask]
        assert value.dtype == torch.float32
        bt_per_number = 4
        for s in value.shape:
            bt_per_number *= s
        bt += bt_per_number
bt_per_second = bt / sample_frame_num * data_frequency
result_num_avg = result_num_sum / sample_frame_num
print(f"UniV2X Instance BPS: {bt_per_second}, Result Per Frame: {result_num_avg}")

# # CoopTrack BPS
# bt = 0
# result_num_sum = 0
# inf_keys = [
#     'query_feats',
#     'cache_motion_feats',
#     'ref_pts',
# ]  # Remove 'query_embeds', as it can be calculated from ref_pts
# for _, sample in tqdm(sample_list, desc="Calculating CoopTrack Instance BPS"):
#     with open(os.path.join(cooptrack_instance_path, sample + ".pkl"), 'rb') as f:
#         data = pickle.load(f)

#     result_num_sum += data['ref_pts'].shape[0]
#     for key in inf_keys:
#         value = data.get(key)
#         assert value.dtype == torch.float32
#         bt_per_number = 4
#         for s in value.shape:
#             bt_per_number *= s
#         bt += bt_per_number
# bt_per_second = bt / sample_frame_num * data_frequency
# result_num_avg = result_num_sum / sample_frame_num
# print(f"CoopTrack Instance BPS: {bt_per_second}, Result Per Frame: {result_num_avg}")
