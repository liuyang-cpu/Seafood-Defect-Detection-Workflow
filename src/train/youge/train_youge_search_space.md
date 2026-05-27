# Youge 训练调参搜索空间说明

这份文档用于说明 `youge` 检测模型当前训练基线、推荐的调参方向，以及建议的搜索空间。

目标不是盲目追求更高的通用 `mAP`，而是优先改善当前最明显的问题：

- 相邻帧中同一对象在被截断后容易被分错类
- 上下边缘区域的目标稳定性不够
- 完整目标和截断目标之间的类别一致性不够好

## 当前基线

当前 `version003` 的实际训练参数，来自：

- [train_youge.py](E:/deepLearning/git/ultralytics-8.3.163-2/ultralytics-8.3.163/src/train/youge/train_youge.py)
- [train_youge.json](E:/deepLearning/git/ultralytics-8.3.163-2/ultralytics-8.3.163/src/train/youge/train_youge.json)
- [args.yaml](E:/deepLearning/git/ultralytics-8.3.163-2/ultralytics-8.3.163/src/train/youge/runs/train/youge_yolo11n_version003/args.yaml)

核心基线如下：

- `epochs = 30`
- `imgsz = 640`
- `batch = 4`
- `workers = 0`
- `amp = false`
- `optimizer = auto`
- `mosaic = 0.0`
- `mixup = 0.0`
- `copy_paste = 0.0`
- `degrees = 0.0`
- `translate = 0.1`
- `scale = 0.5`
- `fliplr = 0.5`
- `hsv_h = 0.015`
- `hsv_s = 0.7`
- `hsv_v = 0.4`
- `erasing = 0.4`

## 现状判断

这组参数有两个明显特点：

1. 训练时长偏短

- `30 epoch` 对当前任务偏少
- 模型还没有充分学习到“完整目标”和“边缘截断目标”的稳定对应关系

2. 形态增强偏强

- `translate = 0.1`
- `scale = 0.5`
- `erasing = 0.4`

这些增强对通用检测任务未必有问题，但对你这个“相机固定、传送带稳定、目标方向稳定”的任务来说，可能会制造过多不真实的截断形态，反而放大误分类。

## 调参原则

针对当前问题，建议遵循这几个原则：

### 1. 先提高分辨率和训练时长

优先级最高的是：

- 提高 `imgsz`
- 增加 `epochs`
- 打开 `amp`

原因：

- 截断目标可见信息少，对分辨率更敏感
- 当前训练轮数偏少，模型还没充分收敛
- `amp` 通常能让更高分辨率训练更可承受

### 2. 收窄几何增强

建议把这几项收窄：

- `mosaic`
- `translate`
- `scale`
- `erasing`

原因：

- 你的真实场景几何变化不大
- 过强增强会让模型学习到不真实形态
- 这类任务更怕“错误的形态先验”

### 3. 不要一开始就放开所有增强

以下参数建议先固定：

- `mixup = 0`
- `copy_paste = 0`
- `degrees = 0`
- `shear = 0`
- `perspective = 0`
- `flipud = 0`

原因：

- 这些增强更可能改变目标形态语义
- 当前最需要的是保持类别定义稳定，而不是制造更多复杂形态

## 推荐搜索空间

完整搜索空间见：

- [train_youge_search_space.json](E:/deepLearning/git/ultralytics-8.3.163-2/ultralytics-8.3.163/src/train/youge/train_youge_search_space.json)

这里先给出最推荐的第一阶段搜索范围。

### 第一阶段小网格

- `epochs`: `60`, `90`, `120`
- `imgsz`: `640`, `768`, `896`
- `batch`: `4`, `8`
- `amp`: `true`
- `mosaic`: `0.0`, `0.1`, `0.2`
- `translate`: `0.02`, `0.05`, `0.08`
- `scale`: `0.15`, `0.25`, `0.35`
- `erasing`: `0.0`, `0.1`, `0.2`

这组参数的目的不是一次性搜全，而是快速判断：

- 更高分辨率是否明显改善截断目标
- 更长训练是否明显改善跨帧类别一致性
- 收窄增强是否能减少截断误分类

## 推荐优先级

建议按下面顺序调：

1. `imgsz`
2. `epochs`
3. `amp`
4. `mosaic`
5. `translate`
6. `scale`
7. `erasing`
8. `lr0`
9. `weight_decay`

原因：

- 前 3 项通常决定上限
- 中间几项主要决定“是否学偏”
- 学习率和正则可以放到第二阶段再细调

## 建议的首轮实验

如果你现在只想先试几组，不想直接跑大搜索，我建议先做这 4 组：

### 实验 A

- `epochs = 90`
- `imgsz = 768`
- `batch = 4`
- `amp = true`
- `mosaic = 0.0`
- `translate = 0.05`
- `scale = 0.25`
- `erasing = 0.1`

### 实验 B

- `epochs = 90`
- `imgsz = 896`
- `batch = 4`
- `amp = true`
- `mosaic = 0.0`
- `translate = 0.05`
- `scale = 0.25`
- `erasing = 0.1`

### 实验 C

- `epochs = 120`
- `imgsz = 768`
- `batch = 8`
- `amp = true`
- `mosaic = 0.1`
- `translate = 0.05`
- `scale = 0.25`
- `erasing = 0.1`

### 实验 D

- `epochs = 120`
- `imgsz = 768`
- `batch = 4`
- `amp = true`
- `mosaic = 0.0`
- `translate = 0.02`
- `scale = 0.15`
- `erasing = 0.0`

## 如何判断实验好坏

不要只看总 `mAP`，建议同时看：

- 常规 `precision / recall / mAP`
- 你当前 `opt` 报告里的类别级表现
- 相邻帧中同一对象是否更少出现 `0 <-> 3` 这种错类
- 上下边缘截断目标在预测后的稳定性

如果一组参数：

- 总 `mAP` 变化不大
- 但截断目标错类明显下降

那它对你这个任务仍然可能是更优的。

## 后续建议

当前训练脚本 [train_youge.py](E:/deepLearning/git/ultralytics-8.3.163-2/ultralytics-8.3.163/src/train/youge/train_youge.py) 只暴露了少量训练参数。

如果后面要真正做系统性搜索，建议下一步：

1. 把更多 Ultralytics 训练参数透传到 `train_youge.py`
2. 建一个小型实验表，记录每次训练的：
   - 参数
   - 总体指标
   - 截断目标专项表现
3. 先跑第一阶段小网格
4. 再围绕最优区域做二次细化
