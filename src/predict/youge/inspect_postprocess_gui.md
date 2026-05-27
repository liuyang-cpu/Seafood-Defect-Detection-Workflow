# Postprocess GUI

`inspect_postprocess_gui.py` 用来直接查看某个 `predict` 输出目录里的图片、框和后处理判定参数，不生成额外结果文件。

## 启动

```powershell
python src/predict/youge/inspect_postprocess_gui.py
```

启动后：

1. 点击 `选择 Predict 目录`
2. 选择一个类似 `src/predict/youge/runs/predict/youge_predict_version003_pp003` 的目录
3. 左侧选择图片
4. 中间查看图片和框
5. 在图片下方框列表中选择某个框
6. 右侧查看当前框的参数比较和规则判定

## 依赖文件

所选目录下至少需要存在：

- `phase2_postprocess_summary.json`
- `run_config.json`

## 右侧展示顺序

右侧详情区按下面顺序展示：

1. `参数比较阈值`
2. `base_conf -> final_conf`
3. `命中规则` 和 `判定路径`
4. `horizontal 规则`
5. `vertical 规则`

## 参数比较阈值

详情区最上面固定展示这 4 个阈值，便于你直接把当前框的实际值和本次 recipe 的阈值对照起来：

- `horizontal_flat_ratio_threshold`
- `horizontal_edge_span_threshold`
- `vertical_intact_aspect_ratio_threshold`
- `vertical_edge_span_threshold`

这 4 个值来自当前 `predict` 目录下 `run_config.json` 的 `postprocess_rules`。

## 比较规则

### horizontal

GUI 会展示这些实际值：

- `horizontal_rule_edge_type`
- `horizontal_top_gap`
- `horizontal_bottom_gap`
- `horizontal_min_edge_gap`
- `horizontal_width_height_ratio`
- `horizontal_edge_span_score`

当前规则判断顺序：

1. `horizontal_min_edge_gap <= horizontal_edge_touch_px`
2. `horizontal_width_height_ratio >= horizontal_flat_ratio_threshold`
3. `horizontal_edge_span_score > horizontal_edge_span_threshold`

三条都满足时，命中 `horizontal_edge_penalty`。

### vertical

GUI 会展示这些实际值：

- `edge_type`
- `phase2_decision`
- `phase2_reason`
- `phase2_aspect_ratio`
- `phase2_edge_span_score`

当前规则判断顺序：

1. 先判断是否为 `top` / `bottom` 贴边框
2. 如果 `phase2_aspect_ratio >= vertical_intact_aspect_ratio_threshold`，判为 `intact`
3. 否则如果 `phase2_edge_span_score >= vertical_edge_span_threshold`，判为 `defective`
4. 否则判为 `intact`

## 命中关系

当前 GUI 和后处理逻辑保持一致：

- 先判断 `horizontal`
- 如果已经命中 `horizontal_edge_penalty`，就不再对该框应用 `vertical` 降分
- 如果 `horizontal` 未命中，再进入 `vertical` 判定

## 说明

- GUI 只做查看，不会改动原始预测结果
- 主要用于人工复核单张图、单个框在后处理中的判定过程
