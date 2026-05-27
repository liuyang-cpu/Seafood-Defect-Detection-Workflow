# Prompt: YOLO Official Parameter Inventory And Explanation

你是一名熟悉 Ultralytics YOLO 官方文档、训练配置和超参数搜索的高级算法工程师。请围绕“YOLO 官方文档中的参数”做一份系统说明，要求先把参数完整罗列出来，再解释每个参数的作用。

## Source Priority

请严格按下面优先级取材：

1. Ultralytics 官方文档中的参数说明口径
2. 当前仓库中的本地配置文件与源码，用来交叉核对参数名、默认值和任务归属

本地核对文件：

- `ultralytics/cfg/default.yaml`
- `ultralytics/cfg/__init__.py`
- `ultralytics/utils/tuner.py`
- `datasets/data/youge/classes.txt`
- `src/train-circle/README.md`
- `src/train-circle/docs/planner_rules.md`
- `src/train-circle/docs/llm_planner_io_spec.md`
- `src/train-circle/plans/planner_runs/*/planner_payload.json`
- `src/train-circle/plans/planner_runs/*/planner_response.json`
- `src/predict/youge/runs/predict/youge_predict_version008_pp003`

如果官方文档口径与当前仓库参数存在差异，请：

- 优先保留“当前仓库版本实际存在的参数名”
- 明确注明“文档口径”和“当前仓库版本实现”之间的差别

## Project Context

这不是一个通用 COCO 检测任务，而是一个具有明确业务背景的海鲜缺陷目标检测项目。请在整个分析过程中显式纳入以下背景，不要把它当成普通的通用检测数据集来讨论。

### 数据集与类别

根据 `datasets/data/youge/classes.txt`，当前类别包括：

- `broken`
- `normal`
- `muddy`
- `empty`

请把这个任务理解为“海鲜目标状态/缺陷检测”，而不是纯粹的单类别目标检测。

### 场景特征

结合 `src/train-circle` 中的任务描述与 `src/predict/youge/runs/predict/youge_predict_version008_pp003` 中的推断结果，可推断当前场景具有以下特点：

- 相机固定
- 场景几何变化较小
- 背景较单一、较浅
- 目标沿扫描/传送方向连续出现
- 图像中经常同时出现多个海鲜目标
- 目标存在靠近边缘、部分截断、跨扫描线附近出现的情况
- 小目标与边缘目标更敏感

### 当前业务问题

请明确纳入 `src/train-circle` 中已经反复提到的问题：

- 相邻帧截断框误分类
- 边缘目标稳定性不足
- 小目标更敏感
- 增强不宜过强
- 优先解决业务问题，而不是只追求总体 mAP50-95

### 从推断结果中应当吸收的观察

结合 `youge_predict_version008_pp003` 里的预测结果，请把以下观察作为分析背景：

- 推断结果中四个类别都出现过，因此这是多类别业务问题，不应只围绕单一类别设计搜索空间
- 同一批结果里既有大量非空标签，也有相当数量的空标签文件，说明“空检测/漏检”和“有目标帧的稳定识别”都值得关注
- 在已观察样本中，`broken / normal / muddy / empty` 之间存在潜在混淆风险，尤其是在目标靠边、目标较小、局部截断时
- 某些 `empty` 预测置信度相对不高，因此在参数分析中需要考虑“低置信度类别混淆”而不只是框是否打中
- 不要仅因为某组参数带来更高总 mAP，就默认它更适合该业务；如果它可能伤害边缘稳定性、截断识别或类别一致性，应明确指出

## Core Task

请完成以下目标：

1. 先按官方文档体系，把 YOLO 参数完整罗列出来
2. 再介绍每个参数各自的作用
3. 区分哪些参数属于训练主线，哪些只是验证、预测、导出或跟踪阶段参数
4. 对训练参数补充“是否适合做 Optuna 搜索”的工程判断
5. 根据筛选出的训练调优核心参数，进一步落地更新 `src/train-optuna/youge_optuna_config.json`
6. 在所有参数分析和搜索建议中，结合上述海鲜缺陷检测业务背景进行针对性判断

## Output Order

请严格按照下面顺序输出，不要跳步骤：

### 1. 官方参数总览

先按官方文档常见栏目把参数罗列出来。建议按下面分组：

- Train settings
- Hyperparameters
- Val/Test settings
- Predict settings
- Visualize settings
- Export settings
- Tracker settings
- Task-specific settings

每个分组先给一张参数清单表，表中必须包含：

- 参数名
- 默认值
- 类型
- 所属栏目
- 是否属于训练调优核心参数：`是 / 否`

### 2. 训练调优核心参数清单

在官方参数总览之后，单独筛出“正常 YOLO 训练调优时最相关”的参数。

这里的训练调优核心参数，优先包含：

- 会直接影响训练计划
- 会直接影响优化过程
- 会直接影响数据增强
- 会直接影响显存、吞吐或训练稳定性

请在这一节明确给出：

- 训练调优核心参数总数
- 你的筛选口径
- 哪些参数虽然官方存在，但不属于训练调优核心

### 3. 每个参数的作用说明

对所有训练调优核心参数逐个解释，格式统一。

每个参数都要说明：

- 参数名
- 默认值
- 作用
- 它主要影响什么：
  - 训练时长
  - 收敛速度
  - 最终精度
  - 数据增强强度
  - 显存/吞吐
  - 推理表现间接影响
- 调大通常会怎样
- 调小通常会怎样
- 常见风险或副作用

### 4. Detect 任务视角的说明

请专门从 detect 任务角度补充判断：

- 哪些参数最值得优先关注
- 哪些参数通常应先固定
- 哪些参数虽然官方提供，但对 detect 任务帮助通常不大
- 哪些参数可能影响 `broken / normal / muddy / empty` 之间的类别混淆
- 哪些参数可能影响边缘截断目标、相邻帧稳定性和小目标表现

### 5. Optuna 搜索建议

最后再补一节工程建议：

- 哪些参数适合第一阶段 Optuna 搜索
- 哪些参数更适合先固定
- 哪些参数更适合离散搜索
- 哪些参数更适合连续搜索
- 哪些参数如果调得过强，可能会伤害海鲜缺陷检测场景中的边缘稳定性和截断识别

### 6. 写回 JSON 配置

在完成参数分析后，请继续执行一项落地动作：根据你筛选出的训练调优核心参数，补充并更新下面这个文件：

- `src/train-optuna/youge_optuna_config.json`

更新时请遵守以下规则：

- 保持原有 JSON 结构不被破坏
- 优先更新这些字段：
  - `enabled_params`
  - `fixed_overrides`
  - 如有必要，可调整 `n_trials`
  - 如有必要，可调整 `metric`
  - 如有必要，可调整 `sampler`
- `enabled_params` 应只保留你认为适合当前 detect 任务、且适合第一阶段 Optuna 搜索的核心参数
- 不要把明显属于 `predict / export / track / visualize` 的参数写入 `enabled_params`
- 如果某些参数虽然是训练参数，但你判断第一阶段不应搜索，请不要写入 `enabled_params`
- 如果当前 JSON 中已有不合理的参数组合，请直接修正，并说明原因
- 如果你建议保守版搜索空间，请让 `enabled_params` 更收敛；如果你同时给出激进版方案，请把激进版作为说明，不要直接覆盖保守版默认配置
- 写回 JSON 后，请额外输出一段“本次对 JSON 做了哪些修改”的说明

## Additional Constraints

- 必须使用 `default.yaml` 中实际存在的原始参数名
- 不要凭空创造不存在的参数
- 不要把 `predict / export / track` 参数混进训练核心参数里
- 先“罗列参数”，再“解释作用”，顺序不能反
- 如果某些参数仅对分类、分割、姿态等任务有效，必须明确标出
- 如果某些参数在官方文档中出现，但当前仓库版本行为不同，必须标记差异
- 如果你修改了 `src/train-optuna/youge_optuna_config.json`，必须说明修改依据来自哪些核心参数判断
- 不能脱离海鲜缺陷检测这个具体业务去泛化讨论
- 必须显式考虑：类别混淆、边缘截断、相邻帧稳定性、小目标敏感性、增强过强的副作用

## Output Style

- 用中文输出
- 结构清晰
- 先总表，后细讲
- 不要只讲概念，必须落到每个具体参数
