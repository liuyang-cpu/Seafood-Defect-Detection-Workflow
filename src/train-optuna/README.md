# Train Optuna

这里放一套基于 `Optuna` 的超参数搜索实现，目标是复用现有 `src/train/youge/train_youge.py` 的训练入口，而不是再复制一套 YOLO 训练逻辑。

这样做的好处：

- 继续沿用当前数据集解析和运行时 `dataset.yaml` 生成逻辑
- 继续沿用 `versionXXX` 模型归档和 `run_summary.json`
- 继续把真实训练结果落到 `src/train/youge/runs/train`
- 把 Optuna 自己的 study、trial 配置、trial 日志、报表单独放在 `src/train-optuna/runs/studies`

## 文件

- `run_youge_optuna_search.py`
  - Optuna 搜索入口
  - 每个 trial 会调用一次现有 `train_youge.py`
- `report_youge_optuna_study.py`
  - 读取 study 数据库和 trial 结果，生成汇总报表
- `youge_optuna_config.json`
  - 默认搜索配置
- `common.py`
  - 公共路径、JSON、metric、override 处理

## 依赖

必须使用能够正常执行 `src/train/youge/train_youge.py` 的同一个 Python 环境安装和运行 Optuna。比如本机当前验证可用的 CUDA 环境是 `yolo-pip`：

```bash
conda activate yolo-pip
python -m pip install optuna
python -c "import optuna; from ultralytics import YOLO; import torch; print('ready', torch.cuda.is_available())"
```

如果你还没装 `matplotlib`，报表 PNG 会跳过，但 CSV/JSON/Markdown 仍会生成。

## 直接运行

在仓库根目录执行：

```bash
python src/train-optuna/run_youge_optuna_search.py
```

如果想先改 trial 数量：

```bash
python src/train-optuna/run_youge_optuna_search.py --n-trials 24
```

如果想换 study 名称：

```bash
python src/train-optuna/run_youge_optuna_search.py --study-name youge_optuna_stage2
```

## 生成报表

```bash
python src/train-optuna/report_youge_optuna_study.py --study-name youge_optuna_tpe_v1
```

## 目录产物

执行后主要会得到两类产物：

- `src/train/youge/runs/train/...`
  - 真实 YOLO 训练目录
  - 包含 `weights/`、`results.csv`、`run_summary.json`
- `src/train-optuna/runs/studies/<study_name>/...`
  - `study.db`
  - `resolved_config.json`
  - `best_trial.json`
  - `best_trial_config.json`
  - `trials/trial_0000/...`
  - `reports/latest/...`

## 当前实现边界

这版是“Optuna 负责采样，训练仍然整轮跑完”的模式。

也就是说：

- 已经是标准的 Optuna 超参搜索
- 但还没有做 epoch 级别的 `trial.report(...)` / `trial.should_prune()`
- 所以 `MedianPruner` 这类早停裁剪目前没有真正发挥作用

如果你下一步要做更像生产版的搜索，我建议再补一层：

1. 直接接入 Ultralytics trainer callback
2. 每个 epoch 把 `mAP50-95` 回传给 Optuna
3. 让 Optuna 中途 prune 差的 trial

这一步需要改成“同进程训练 + callback 汇报”，复杂度会明显高于当前版本。

## 使用建议

- 单卡 `3080 Ti 12GB` 先保持 `Optuna n_jobs=1`
- 不要并发跑多个 trial
- 第一轮先把 `n_trials` 控制在 `12~24`
- 优先看 `best_trial.json` 和 `reports/latest/summary.csv`
- 如果显存不稳，先在 `youge_optuna_config.json` 里把 `batch` 固定住
