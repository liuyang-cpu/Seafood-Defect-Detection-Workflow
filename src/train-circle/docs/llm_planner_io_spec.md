# LLM Planner 输入输出协议

## 1. 目的

这份协议用于约束 `llm_planner.py` 和后续 `llm_plan_materializer.py` 之间的数据边界。

目标：

- 不直接修改现有训练脚本
- 先定义稳定的数据协议
- 后续无论接 OpenAI API 还是本地 mock，都走同一套结构

---

## 2. 设计原则

- LLM 只负责“实验建议”，不直接生成最终训练代码
- 所有可执行配置都必须经过本地校验层
- 输入只发送实验摘要，不发送原始图像或长日志
- 输出必须是结构化 JSON，不接受自由文本
- 所有参数必须限制在本地白名单和搜索空间内

---

## 3. LLM 请求载荷

建议由 `llm_planner.py` 组装一个 JSON 请求对象，再将其作为模型输入上下文。

### 3.1 顶层结构

```json
{
  "task_context": {},
  "baseline_run": {},
  "search_space": {},
  "history_summary": [],
  "budget_constraints": {},
  "planner_rules": {}
}
```

### 3.2 `task_context`

描述任务背景和优化目标。

```json
{
  "project": "youge",
  "model_family": "YOLO11n",
  "dataset_name": "youge",
  "problem_summary": [
    "相邻帧截断框误分类",
    "边缘目标稳定性不足",
    "小目标更敏感"
  ],
  "primary_objective": "优先改善截断目标和边缘目标的检测稳定性，同时关注总体 mAP50-95",
  "notes": [
    "相机固定",
    "场景几何变化较小",
    "增强不宜过强"
  ]
}
```

### 3.3 `baseline_run`

这里不要直接取“当前 train_youge.json”，应绑定到一个已完成实验。

默认语义：

- 如果没有显式指定 `baseline_version`
- 就取历史中效果最好的已完成训练产物作为 baseline
- 默认按 `best_metrics.map50_95` 选择

推荐来源：

- `run_summary.json`
- 或 `summary.csv` 中人工指定的某个 `version`

### 3.4 `search_space`

建议只暴露本轮允许 LLM 规划的参数。

推荐先开放：

- `imgsz`
- `epochs`
- `mosaic`
- `translate`
- `scale`
- `erasing`
- `amp`

### 3.5 `history_summary`

每一项代表一个已完成实验。建议不要只保留 `summary.csv` 里的少数字段，而是尽量从 `run_summary.json` 恢复完整训练参数。

注意：

- `baseline_run` 已经单独提供
- `history_summary` 默认不再重复包含这条 baseline
- 这里更适合放“除 baseline 之外的其他历史实验”，避免模型把同一条基线看两遍

### 3.6 `budget_constraints`

建议至少包含：

- `max_recommended_runs`
- `max_epochs_per_run`
- `max_imgsz`
- `max_batch`
- `require_non_duplicate_runs`

### 3.7 `planner_rules`

建议至少包含：

- `must_stay_within_search_space`
- `must_not_repeat_history`
- `prefer_conservative_narrowing`
- `focus_on_business_problem_over_raw_map`

---

## 4. LLM 响应结构

LLM 响应必须符合 `llm_planner_response_schema.json`。

```json
{
  "observations": [
    "更高分辨率带来了有限收益，但收益小于预期"
  ],
  "frozen_params": {
    "mixup": 0.0
  },
  "range_updates": {
    "translate": [0.02, 0.05]
  },
  "recommended_runs": [
    {
      "label": "round2_conservative_896",
      "reason": "提高分辨率，同时收窄增强。",
      "overrides": {
        "imgsz": 896,
        "epochs": 90,
        "mosaic": 0.1,
        "translate": 0.02,
        "scale": 0.15,
        "erasing": 0.0,
        "amp": true
      }
    }
  ]
}
```

---

## 5. 本地校验要求

LLM 输出不能直接执行。建议 `llm_plan_materializer.py` 做以下检查：

- 参数名是否全部在白名单中
- 每个值是否落在允许集合内
- 推荐实验数量是否超预算
- 是否与历史实验完全重复
- 是否与本轮输出中的其他推荐重复
- `label` 是否满足文件名安全要求
