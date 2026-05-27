# Train Circle

这里用于放置新的“训练规划链路”，不直接修改现有 `src/train/youge` 训练脚本。

目标：

- 保留原始训练入口和当前训练任务不受影响
- 在独立目录中建设 LLM 辅助训练规划层
- 后续逐步补齐 `mock -> API -> 计划生成 -> 人工审核`

当前建议的职责边界：

- `src/train/youge`
  - 继续负责单次训练、搜索计划生成、计划执行、报告汇总

- `src/train-circle`
  - 负责 LLM 规划协议
  - 负责读取历史实验摘要
  - 负责生成下一轮候选实验建议
  - 负责将建议转换为待审核计划

目录结构：

- `docs/`
  - 协议文档和 schema
- `prompts/`
  - 规划器提示词模板
- `config/`
  - 提供方与模型配置
- `scripts/`
  - 可执行脚本和公共模块
- `examples/`
  - 响应示例和本地测试输入
- `artifacts/`
  - mock 生成的中间产物
- `plans/`
  - LLM 生成的待审核计划和已落盘训练计划

当前核心脚本：

- `scripts/llm_planner_mock.py`
  - 本地 mock 骨架，不调用 API，只验证数据链路
- `scripts/llm_planner.py`
  - 真实 API 调用入口
- `scripts/llm_plan_materializer.py`
  - 本地校验和计划落盘
- `scripts/planner_common.py`
  - 公共逻辑层
