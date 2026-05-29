# LLM Param Space Opt

这里放一套独立于 `src/train-optuna` 的新流程，用于把一轮 Optuna 实验结果交给大模型分析，再产出下一轮搜索空间和 Optuna 配置建议。

目标：

- 不直接耦合旧的 `run_youge_optuna_search.py`
- 复用现有 Optuna study 产物与报表
- 让大模型输出“下一轮搜索空间建议”，而不是几组手工实验
- 所有可执行产物都必须经过本地校验后再落盘

当前最小闭环：

1. 读取指定 Optuna study
2. 生成结构化 payload
3. 调用 LLM 输出下一轮建议
4. 本地校验建议
5. 生成 `next_round/` 目录，其中包含：
   - `youge_optuna_config.json`
   - `train_youge_search_space.json`
   - `manifest.json`

## 目录

- `config/`
  - LLM 提供方、模型、API 地址与密钥入口配置
- `docs/`
  - 响应 schema
- `prompts/`
  - 提示词模板
- `scripts/`
  - payload 生成、LLM 调用、配置落盘脚本

## 常用命令

调用前先配置：

- `src/llm-param-space-opt/config/llm_space_refiner.config.json`
- 以及其中 `api_key_env` 对应的环境变量

例如：

```bash
export DEEPSEEK_API_KEY="your-key"
```

生成 payload 并调用 LLM：

```bash
/home/zxy/anaconda3/envs/yolo-pip/bin/python src/llm-param-space-opt/scripts/llm_refine_optuna_space.py --study-name youge_optuna_tpe_v1
```

只生成 payload，不调用 API：

```bash
/home/zxy/anaconda3/envs/yolo-pip/bin/python src/llm-param-space-opt/scripts/llm_refine_optuna_space.py --study-name youge_optuna_tpe_v1 --dry-run
```

把 LLM 响应落成下一轮配置：

```bash
/home/zxy/anaconda3/envs/yolo-pip/bin/python src/llm-param-space-opt/scripts/materialize_next_round.py --response src/llm-param-space-opt/runs/refine_runs/<run_dir>/refiner_response.json
```

## 产物

每次调用 LLM 会生成：

- `runs/refine_runs/refine_<timestamp>/refiner_payload.json`
- `runs/refine_runs/refine_<timestamp>/refiner_response.json`
- `runs/refine_runs/refine_<timestamp>/raw_response.json`
- `runs/refine_runs/refine_<timestamp>/refiner_meta.json`

每次 materialize 会生成：

- `runs/materialized_rounds/round_<timestamp>/youge_optuna_config.json`
- `runs/materialized_rounds/round_<timestamp>/train_youge_search_space.json`
- `runs/materialized_rounds/round_<timestamp>/manifest.json`

## 注意

- 这套流程当前是“半自动闭环”，不会自动启动下一轮训练。
- 默认输入来自 `src/train-optuna/runs/studies/<study_name>/`。
- 输出建议会限制在本地搜索空间与白名单参数内。
