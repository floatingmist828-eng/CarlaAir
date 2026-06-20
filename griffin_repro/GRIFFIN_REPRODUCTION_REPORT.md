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

结论：CoopTrack 是当前唯一未达到论文容差的核心方法。复现成果汇报中应将它列为已真实运行但未完全贴合论文的保留项。

## 验证命令

远端环境检查：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py doctor --json
```

查看论文矩阵覆盖：

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py paper-run-matrix \
  --dataset 50scenes_25m \
  --include-robustness \
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

当前分支已经完成 Griffin-25m 的主要复现闭环：数据转换、官方 checkpoint 加载、baseline、早融合全鲁棒性、晚融合全鲁棒性、CoopTrack baseline 和鲁棒性验证、paper-scope 指标解析、真实远端 A100 实验日志与汇总 artifact。若按“主要融合方式和鲁棒性场景贴合论文”评估，no-fusion、early-fusion、late-fusion 已经可以作为复现成果；若按“所有方法严格完全一致”评估，CoopTrack 仍需继续定位。
