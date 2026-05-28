# Scheduler

这里放一套独立调度器，用来串联：

1. `src/train-optuna/run_youge_optuna_search.py`
2. `src/llm-param-space-opt/scripts/llm_refine_optuna_space.py`
3. `src/llm-param-space-opt/scripts/materialize_next_round.py`

目标是形成一个按轮次运行的闭环：

- 跑一轮 Optuna
- 生成 report / best trial
- 调 LLM 分析结果
- 产出下一轮 Optuna 配置与搜索空间
- 继续下一轮

支持两类停机条件：

- `max_rounds`
- `target_metric_threshold`

## 目录

- `config/`
  - 调度器配置样例
- `scripts/`
  - 调度入口脚本
- `runs/`
  - 每次调度循环的运行记录

## 用法

默认配置运行：

```bash
/home/zxy/anaconda3/envs/yolo-pip/bin/python src/scheduler/scripts/run_optuna_llm_scheduler.py
```

指定配置运行：

```bash
/home/zxy/anaconda3/envs/yolo-pip/bin/python src/scheduler/scripts/run_optuna_llm_scheduler.py --config src/scheduler/config/optuna_llm_cycle.config.json
```

只打印将执行的命令，不真正启动：

```bash
/home/zxy/anaconda3/envs/yolo-pip/bin/python src/scheduler/scripts/run_optuna_llm_scheduler.py --dry-run
```

## 产物

每次循环会在 `src/scheduler/runs/<cycle_name>_<timestamp>/` 下生成：

- `resolved_scheduler_config.json`
- `round_01/round_meta.json`
- `round_02/round_meta.json`
- ...
- `cycle_summary.json`

其中每轮会记录：

- 使用的 Optuna config/search space
- study 名称
- best trial value
- LLM refiner 响应路径
- materialized 下一轮配置路径
