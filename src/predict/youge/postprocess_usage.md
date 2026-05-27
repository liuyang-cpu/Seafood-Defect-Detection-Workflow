# Youge Postprocess 说明

这份说明覆盖以下文件：

- `predict_youge.py`
- `phase2_postprocess_common.py`

## 1. 功能定位

当前 `youge` 的后处理已经并入 `src/predict/youge`。

职责分工：

- `predict_youge.py`
  负责正常预测，并在导出结果时应用后处理规则。
- `phase2_postprocess_common.py`
  负责真正的后处理规则实现和结果导出。

## 2. 当前会真正降分的规则

### 2.1 `horizontal_edge_penalty`

触发条件：

- 框贴上边缘或下边缘
- `w / h >= horizontal_flat_ratio_threshold`
- `horizontal_edge_span_score > horizontal_edge_span_threshold`

命中后：

- `conf = conf * horizontal_penalty_factor`

关键观察量：

- `w / h`
- `top_gap`
- `bottom_gap`
- `min_edge_gap`
- `horizontal_edge_span_score`

### 2.2 `vertical_defective_penalty`

触发逻辑：

- 先只看贴上边缘或下边缘的框
- 如果 `h / w >= vertical_intact_aspect_ratio_threshold`
  则直接视为 `intact`，不降分
- 否则继续计算 `edge span score`
- 当 `edge span score >= vertical_edge_span_threshold` 时视为 `defective`

命中后：

- `conf = conf * vertical_defective_penalty_factor`

关键观察量：

- `edge_type`
- `h / w`
- `edge_span_score`

补充：

- `vertical_aspect_ratio_override` 和 `vertical_edge_span_ratio` 只是判定过程，不是惩罚规则。
- 同一个框可能同时命中两条惩罚规则，此时两个倍率会连续相乘。

## 3. 复核图导出

命中惩罚规则时会额外导出复核图：

- 输出目录：`<save_dir>/penalty_hits/`
- 子目录：
  - `horizontal_edge_penalty/`
  - `vertical_defective_penalty/`

每张图会展示：

- 被惩罚的框
- 命中的规则名
- 该次命中的实际参数值
- 对应阈值和惩罚倍率

## 4. 推荐流程

1. 正常运行 `predict_youge.py`
2. 查看预测目录中的 `labels/`、渲染图和 `phase2_postprocess_summary.json`

## 5. 运行示例

```powershell
E:\Users\liuyang\anaconda3\envs\yolo\python.exe .\src\predict\youge\predict_youge.py
```

脚本会默认读取同目录下的 `predict_youge.json` 配置文件。
