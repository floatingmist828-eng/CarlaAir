# Griffin Paper Reproduction Report

本报告汇总当前分支对 Griffin 论文和 upstream `wang-jh18-SVM/Griffin` 的复现进展。复现工作隔离在 `griffin_repro/` 下，不依赖 CarlaAir 原自动驾驶代码路径。

## 复现范围

- 代码来源：`wang-jh18-SVM/Griffin`，本地 manifest 记录 upstream HEAD `9c02ba4a37201edfc2b95ddbcdc2ff9aff47e7f4`。
- 已真实运行数据集：`Griffin-50scenes-25m`，当前远端可用官方验证子集为 1490 frames。
- 已覆盖场景：baseline、communication latency 100/200/300/400 ms、packet loss 0.1/0.2/0.3/0.4/0.5、translation error 0.5/1.0/1.5/2.0/2.5 m、rotation error 1/2/3/4/5 deg。
- 已覆盖可运行融合方式：`0-no fusion`、`1-early fusion`、`2b1-cooptrack`、`3-late fusion`。
- 论文矩阵中无 released config/checkpoint 的方法已经在 `manifest.json` 和 `paper-run-matrix` 中保留为 paper-result-only 行，但未伪造实验结果。

## 指标口径

论文 `docs/detailed_results.csv` 对齐的是 car 类 paper-scope 指标：检测 AP 来自 car class AP，跟踪 AMOTA 来自 car class AMOTA。官方 evaluator 的三类 aggregate mAP/AMOTA 不应直接和论文表格比。当前报告统一使用 `paper_tolerance=0.02` 判断是否贴合论文。

## 基线结果

远端汇总文件：`griffin_repro/artifacts/logs/official_25m_baseline_all_20260620_summary.json`。

| Method | Actual AP | Paper AP | Actual AMOTA | Paper AMOTA | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0-no fusion | 0.374769 | 0.375 | 0.365143 | 0.365 | match |
| 1-early fusion | 0.607000 | 0.607 | 0.670000 | 0.670 | match |
| 2b1-cooptrack | 0.420000 | 0.479 | 0.453000 | 0.488 | below paper |
| 3-late fusion | 0.377000 | 0.378 | 0.379000 | 0.377 | match |

结论：no-fusion、early-fusion、late-fusion 基线已经贴合论文；CoopTrack baseline 仍低于论文，差值约 AP `-0.059`、AMOTA `-0.035`。

## 鲁棒性结果

### Early Fusion

远端汇总文件：`griffin_repro/artifacts/logs/official_25m_early_all_20260620_summary.json`。

`1-early fusion` baseline 加 19 个鲁棒性条件共 20 行全部通过 `0.02` 容差：AP `20/20` 通过，AMOTA `20/20` 通过。代表值如下：

| Condition | Actual AP | Paper AP | Actual AMOTA | Paper AMOTA |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.607 | 0.607 | 0.670 | 0.670 |
| latency 400 ms | 0.420 | 0.420 | 0.450 | 0.450 |
| packet loss 0.5 | 0.530 | 0.530 | 0.611 | 0.611 |
| translation error 2.5 m | 0.416 | 0.416 | 0.454 | 0.455 |
| rotation error 5 deg | 0.442 | 0.442 | 0.484 | 0.484 |

### Late Fusion

远端汇总文件：`griffin_repro/artifacts/logs/official_25m_late_robust_all_20260620_summary.json`。

`3-late fusion` 19 个鲁棒性条件全部通过 `0.02` 容差：AP `19/19` 通过，AMOTA `19/19` 通过。代表值如下：

| Condition | Actual AP | Paper AP | Actual AMOTA | Paper AMOTA |
| --- | ---: | ---: | ---: | ---: |
| latency 100 ms | 0.374 | 0.375 | 0.380 | 0.378 |
| latency 400 ms | 0.346 | 0.346 | 0.315 | 0.319 |
| packet loss 0.5 | 0.334 | 0.341 | 0.351 | 0.369 |
| translation error 2.5 m | 0.315 | 0.315 | 0.346 | 0.346 |
| rotation error 5 deg | 0.291 | 0.292 | 0.296 | 0.293 |

Late-fusion latency 配置暴露了 upstream converter 的一个兼容缺口：`det_result_late_fusion.py` 无条件读取 `cfg.drop_prob / cfg.loc_noise_std / cfg.orien_noise_std`，但 latency config 未定义这些字段。本分支已修正为默认 `0.0`，含回归测试。该默认值等价于 latency 场景不额外叠加丢包和位姿噪声。

### CoopTrack

远端汇总文件：`griffin_repro/artifacts/logs/official_25m_cooptrack_robust_all_20260620_summary.json`。

`2b1-cooptrack` baseline 和鲁棒性行都能完整运行，但数值不稳定贴合论文。18 个鲁棒性条件中，AP `11/18` 通过、AMOTA `4/18` 通过、AP+AMOTA 同时通过 `1/18`。baseline 当前为 AP `0.420` vs paper `0.479`，AMOTA `0.453` vs paper `0.488`。

已排查证据：

- cooperative instance-fusion checkpoint md5 与 upstream release 一致：`7e1448188b6e99ca6303575c3466b97f`。
- track-query cache 覆盖 `1490/1490` frames，无缺失 air-token 文件，无 NaN/inf tensor，无负 track id 或帧内重复 id。
- 当前 CoopTrack car 统计为 `TP/FP/FN/IDS = 4685/1293/3611/24`，论文为 `3755/599/4563/2`。当前输出不是漏检，而是 false positives 和 ID switches 明显更多。
- 新增 `audit-cooptrack-gap` 诊断命令后，远端复核文件为 `griffin_repro/artifacts/logs/official_25m_cooptrack_gap_audit_20260620.json`。该文件确认：query 时序完整，`expected_sequence_count=1490`、`missing_expected_count=0`、`first_missing_index=null`；query cache 中 active queries 为 `8004`，其中 `>=0.35` 为 `7047`、`>=0.4` 为 `6822`；tracking result 的 `track_ids` 无负数、无帧内重复，`unique_id_count=216`、每帧有效 id 平均 `4.6691`。因此当前差距不来自 query 文件缺失或明显 track-id 写坏。
- score threshold 离线扫描显示，提高 detection score threshold 会继续降低 car AP：`thresh_0` car AP `0.4203`，`0.3` 为 `0.3773`，`0.5` 为 `0.3659`，`0.9` 为 `0.2982`。因此当前 CoopTrack 偏差也不是简单提高 score threshold 就能修复。
- 当前配置阈值已复核：`score_thresh=0.4`、`filter_score_thresh=0.35`、`train_gt_iou_threshold=0.3`、`bbox_coder=NMSFreeCoder(max_num=300)`。下一步应优先检查 cross-agent query matching/补全策略和 evaluator/后处理细节，而不是盲目调高 NMS 或 score 阈值。
- 进一步远端真实实验显示，当前坐标链路不是主要问题。对 drone `ref_pts` 到 vehicle frame 的 4 种矩阵候选做最近 car GT 距离统计，当前代码路径 `inv(info['vehLidar2airLidar_rt']) @ point` 最优：active query `8004`，最近 car GT XY 距离 mean `2.66m`、p50 `1.246m`、p90 `5.433m`，其他候选 mean `9.35m-15.36m`。
- 进一步远端真实实验排除了简单 query 过滤和 checkpoint 别名问题。query score 过滤 `>=0.4` 得到 AP `0.4111`、AMOTA `0.4315`，`>=0.7` 得到 AP `0.3882`、AMOTA `0.4247`，均低于 baseline；下载并测试顶层 `cooperative.pth` 后得到 AP `0.4050`、AMOTA `0.4267`，也低于当前 nested `cooperative/instance_fusion/iter_33024.pth`。
- 进一步远端真实实验确认 val order 没有异常：`griffin_infos_val.pkl` 为 10 scenes x 149 frames，scene 内 frame step 连续，timestamp 排序与现有顺序一致。tracker 参数微调只能小幅改善：`score_thresh=0.7` 当前最好，AP `0.4228`、AMOTA `0.4599`；`score_thresh=0.9` 提高 AP 到 `0.4246` 但 AMOTA 降到 `0.4313`；`miss_tolerance=1/3` 未能闭合论文差距。

结论：CoopTrack 是当前唯一未达到论文容差的核心方法。复现成果汇报中应将它列为已真实运行但未完全贴合论文的保留项；目前证据更像是 released checkpoint/code/evaluator 或论文内部后处理细节存在差异，而不是 25m validation 子集缺文件、坐标矩阵、query 时序或简单阈值问题。

## 仍未闭合的论文内容

- `50scenes_25m`：当前远端不需要把原始大包全部补齐才能复现关键闭环。已 materialize 的 25m validation 子集可支撑 1490-frame 官方评估、no-fusion/early-fusion/late-fusion/CoopTrack baseline 和鲁棒性实验；但原始 full package 清单仍是 `4/15` 完整，25m checkpoint 清单是 `4/5` 完整，缺 `drone-side.pth`，因此不能表述为 25m 全量包已完整下载。
- `50scenes_40m`、`50scenes_55m`、`100scenes_random`：论文矩阵和命令映射已覆盖，且本分支已补齐 Hugging Face 数据包/checkpoint 静态清单。数据包体量分别约为 40m `203.25 GB`、55m `192.91 GB`、random `403.92 GB`；checkpoint 分别约为 40m `1.54 GB`、55m `0.88 GB`、random `1.54 GB`。远端当前尚未落盘这些大包，因此还不能声称真实复现。
- `2a1-v2x-vit`、`2a2-where2comm`、`2b2-univ2x`：论文结果已保留在矩阵中，但当前 zip/upstream checkout 下没有完整可直接运行的 config/checkpoint 闭环；不能伪造为已跑通。
- upstream GitHub `main` 已核对为 `9c02ba4a37201edfc2b95ddbcdc2ff9aff47e7f4`，与 manifest 一致。README 最新消息提到 55m subset/checkpoint、UniV2X pretrained model 和 robustness config 已发布，但仓库页面未提供 GitHub Releases；远端当前仍需从数据源补齐实际数据包和 checkpoint 后才能启动这些行。
- `BPS/FPS`：指标体系已建模，官方 `tools/analysis_tools/compute_BPS.py` 存在，但脚本硬编码路径且 CoopTrack BPS 段为注释状态；未发现官方 `compute_FPS.py`。因此 AP/AMOTA 已真实验证，BPS/FPS 还需要统一硬件测速脚本补齐。

## 验证命令

远端环境检查：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py env-check --strict --json
```

查看论文矩阵覆盖：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py paper-run-matrix \
  --dataset 50scenes_25m \
  --include-robustness \
  --json
```

查看待补 checkpoint 清单：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py checkpoint-packages \
  --dataset 50scenes_55m \
  --json
```

复查 CoopTrack gap audit：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py audit-cooptrack-gap \
  --result-pkl griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls/results-06201945.pkl \
  --query-dir griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query \
  --ann-file griffin_repro/official/data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val.pkl \
  --eval-dir griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls/json_output/Sat_Jun_20_19_49_34_2026 \
  --config griffin_repro/official/projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py \
  --json
```

复查 late-fusion 汇总：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py summarize-official-log \
  --log griffin_repro/artifacts/logs/official_25m_late_robust_packet_loss_0.5_20260620_214500.log \
  --dataset 50scenes_25m \
  --method "3-late fusion" \
  --condition-id packet_loss_0.5 \
  --paper-tolerance 0.02 \
  --json
```

## 当前结论

当前分支已经完成 Griffin-25m 的主要复现闭环：数据转换、官方 checkpoint 加载、baseline、早融合全鲁棒性、晚融合全鲁棒性、CoopTrack baseline 和鲁棒性验证、paper-scope 指标解析、真实远端 A100 实验日志与汇总 artifact。若按“主要融合方式和鲁棒性场景贴合论文”评估，no-fusion、early-fusion、late-fusion 已经可以作为复现成果；若按“所有方法严格完全一致”评估，CoopTrack、25m 全量原始包与缺失 drone checkpoint、40m/55m/random 数据与 checkpoint、V2X-ViT/Where2Comm/UniV2X 可运行闭环、BPS/FPS 统一测速仍需继续补齐。
