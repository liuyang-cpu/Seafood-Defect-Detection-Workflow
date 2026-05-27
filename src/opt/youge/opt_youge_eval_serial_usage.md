# Youge Opt Eval 配置说明

这份说明覆盖以下脚本和配置：

- `opt_youge_eval_serial.py`
- `opt_youge_eval.py`
- `opt_youge_eval.json`

## 1. 输出行为

当前 `opt_youge_eval_serial.py` 会：

- 按多个 `conf_threshold` 逐个评估
- 只保留最终汇总报告目录
- 在汇总目录下生成：
  - `index.html`
  - `summary.json`
  - `summary.csv`
  - `_summaries/`（每个阈值的中间 summary.json）

不会再额外保留每个阈值的独立 HTML 明细目录。

## 2. `--conf-thresholds` 支持的写法

### 2.1 显式枚举

```powershell
--conf-thresholds 0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85
```

### 2.2 区间写法

```powershell
--conf-thresholds 0.70-0.85
```

含义：

- 默认按 `0.01` 递增
- 包含起点和终点

上面这条会展开成：

`0.70, 0.71, 0.72, ..., 0.85`

### 2.3 区间 + 指定步长

```powershell
--conf-thresholds 0.70-0.85:0.02
```

含义：

- 从 `0.70` 到 `0.85`
- 步长 `0.02`

### 2.4 混合写法

```powershell
--conf-thresholds 0.70-0.75,0.80,0.83-0.85
```

## 3. `split` 支持的选项

`opt_youge_eval.py` 和 `opt_youge_eval.json` 里的 `split` 现在支持：

- `train`
- `val`
- `both`

### 3.1 `train`

使用：

- `datasets/data/youge/images/train`
- `datasets/data/youge/labels/train`

### 3.2 `val`

使用：

- `datasets/data/youge/images/val`
- `datasets/data/youge/labels/val`

### 3.3 `both`

同时使用：

- `datasets/data/youge/images/train`
- `datasets/data/youge/images/val`
- `datasets/data/youge/labels/train`
- `datasets/data/youge/labels/val`

注意：

- `labels/train/classes.txt` 和 `labels/val/classes.txt` 已经被自动忽略，不参与逐图标签匹配。
- 如果 `train` 和 `val` 里存在同名图片 stem，评估会报错，因为这种情况下预测结果会互相覆盖，无法可靠评估。

## 4. `max_items` 的真实含义

`opt_youge_eval.json` 里的：

```json
"max_items": 200
```

不是“只在 HTML 里展示 200 张”，而是：

- 只取前 200 张参与整个评估
- TP / FP / FN / Precision / Recall / F1 都只按这 200 张统计

如果要全量评估，建议改成：

```json
"max_items": null
```

或者给一个足够大的值。

## 5. 推荐配置示例

### 5.1 只评估 `val`

`opt_youge_eval.json`

```json
{
  "species": "youge",
  "split": "val",
  "predict_name": "youge_predict",
  "image_dir": null,
  "predict_image_dir": null,
  "gt_labels_dir": null,
  "pred_labels_dir": null,
  "project": "runs/opt",
  "name": "youge_opt_report",
  "conf_threshold": 0.75,
  "iou_threshold": 0.5,
  "max_items": null,
  "exist_ok": true
}
```

### 5.2 同时评估 `train + val`

`opt_youge_eval.json`

```json
{
  "species": "youge",
  "split": "both",
  "predict_name": "youge_predict",
  "image_dir": null,
  "predict_image_dir": null,
  "gt_labels_dir": null,
  "pred_labels_dir": null,
  "project": "runs/opt",
  "name": "youge_opt_report",
  "conf_threshold": 0.75,
  "iou_threshold": 0.5,
  "max_items": null,
  "exist_ok": true
}
```

### 5.3 扫描多个阈值

```powershell
python .\src\opt\youge\opt_youge_eval_serial.py --config .\src\opt\youge\opt_youge_eval.json --conf-thresholds 0.70-0.85
```

### 5.4 指定步长扫描

```powershell
python .\src\opt\youge\opt_youge_eval_serial.py --config .\src\opt\youge\opt_youge_eval.json --conf-thresholds 0.70-0.85:0.02
```

## 6. 汇总页图表说明

汇总页 `index.html` 当前包含：

- 指标折线图：`Precision / Recall / F1`
- 数量折线图：`Pred / TP / FP / FN`

并且会标出每条线的：

- 最高点
- 最低点

用于快速定位最优和最差的 `conf`。
