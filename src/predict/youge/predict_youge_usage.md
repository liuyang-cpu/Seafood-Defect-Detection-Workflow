# Youge Predict 配置说明

这份说明覆盖以下脚本和配置：

- `predict_youge.py`
- `predict_youge.json`

## 1. 输出行为

当前 `predict_youge.py` 的输出目录默认是：

- `src/predict/youge/runs/predict/youge_predict`

每次运行前会先清空这个目录，再重新生成结果。

也就是说当前行为是：

- 不保留上一次预测结果
- 不做增量追加
- 每次都是一份干净的新结果

## 2. `source` 和 `split` 的关系

脚本里同时保留了两个字段：

- `source`
- `split`

当前优先级是：

1. 如果 `source` 有值，优先使用 `source`
2. 如果 `source` 是 `null`，再使用 `split`

因此更推荐的写法是：

```json
"source": null,
"split": "val"
```

这样配置更清晰。

## 3. `split` 支持的选项

`predict_youge.py` 和 `predict_youge.json` 里的 `split` 现在支持：

- `train`
- `val`
- `both`

### 3.1 `train`

使用：

- `datasets/data/youge/images/train`

### 3.2 `val`

使用：

- `datasets/data/youge/images/val`

### 3.3 `both`

同时使用：

- `datasets/data/youge/images/train`
- `datasets/data/youge/images/val`

实现方式不是把所有图片一次性读进内存，而是先生成一个 `source_manifest.txt` 清单文件，再把这个清单交给 YOLO。

## 4. `source` 支持的写法

### 4.1 推荐写法

```json
"source": null,
"split": "val"
```

### 4.2 快捷值

`source` 也兼容以下快捷值：

- `"train"`
- `"val"`
- `"both"`

例如：

```json
"source": "both"
```

也能工作。

但从维护角度更推荐把这些快捷值写在 `split` 里，把 `source` 留给真实路径。

### 4.3 真实路径

如果你确实要手工指定预测源，也可以直接写完整路径：

```json
"source": "E:\\path\\to\\images"
```

这种情况下 `split` 不生效。

## 5. 关键预测参数说明

### 5.0 `imgsz`

推理输入尺寸。

当前推荐写法是：

```json
"imgsz": null,
"version": "version014"
```

当 `imgsz = null` 时，`predict_youge.py` 会按当前 `version` 去对应训练版本的 `run_summary.json` 中读取 `train_config.imgsz`，作为本次推理尺寸。

优先级是：

1. CLI 或 JSON 显式指定的 `imgsz`
2. 当前 `version` 对应训练记录里的 `train_config.imgsz`
3. 兜底默认值 `640`

### 5.1 `conf`

置信度阈值。低于这个分数的预测框会先被过滤掉。

### 5.2 `iou`

这里是预测阶段的 `NMS IoU 阈值`，不是评估脚本里的 GT 匹配 IoU。

作用是控制重复框抑制强度：

- 小一点：更容易压掉重叠框
- 大一点：更容易保留重叠框

### 5.3 `save_txt`

是否把预测框保存成 YOLO txt 标签文件。

### 5.4 `save_conf`

在 txt 标签里是否额外保存置信度分数。

## 6. 推荐配置示例

### 6.1 只跑 `val`

`predict_youge.json`

```json
{
  "weights": null,
  "source": null,
  "split": "val",
  "imgsz": 640,
  "device": "0",
  "conf": 0.75,
  "iou": 0.7,
  "project": "runs/predict",
  "name": "youge_predict",
  "exist_ok": true,
  "save_txt": true,
  "save_conf": true
}
```

### 6.2 只跑 `train`

```json
{
  "weights": null,
  "source": null,
  "split": "train",
  "imgsz": 640,
  "device": "0",
  "conf": 0.75,
  "iou": 0.7,
  "project": "runs/predict",
  "name": "youge_predict",
  "exist_ok": true,
  "save_txt": true,
  "save_conf": true
}
```

### 6.3 同时跑 `train + val`

```json
{
  "weights": null,
  "source": null,
  "split": "both",
  "imgsz": 640,
  "device": "0",
  "conf": 0.75,
  "iou": 0.7,
  "project": "runs/predict",
  "name": "youge_predict",
  "exist_ok": true,
  "save_txt": true,
  "save_conf": true
}
```

## 7. 常用命令

### 7.1 跑 `val`

```powershell
python .\src\predict\youge\predict_youge.py --split val
```

### 7.2 跑 `train`

```powershell
python .\src\predict\youge\predict_youge.py --split train
```

### 7.3 跑 `both`

```powershell
python .\src\predict\youge\predict_youge.py --split both
```

### 7.4 临时改阈值

```powershell
python .\src\predict\youge\predict_youge.py --split both --conf 0.8 --iou 0.6
```
