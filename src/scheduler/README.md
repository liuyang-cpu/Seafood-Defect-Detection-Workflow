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

## LLM 选择

调度器现在支持在 `llm_refiner` 下通过配置直接选择实际使用的 LLM 配置文件，不必手改脚本命令：

```json
{
  "llm_refiner": {
    "enabled": true,
    "config_profile": "deepseek",
    "config_profiles": {
      "deepseek": "src/llm-param-space-opt/config/llm_space_refiner.config.json",
      "freemodel": "src/llm-param-space-opt/config/llm_space_refiner.freemodel.config.json"
    }
  }
}
```

- `config_profile` 选当前要用的 profile
- `config_profiles` 定义 profile 到实际配置文件路径的映射
- 如果 `config_profile` 为空，调度器才会退回到 `llm_refiner.config_path`
- 命令行里如果再传 `--provider/--model/--base-url`，仍然会覆盖配置文件内容

例如切到 freemodel，只需要把：

```json
"config_profile": "freemodel"
```

改掉即可。

## Sampler Seed

调度器支持按 round 自动改变 Optuna sampler 的 seed，避免多轮循环反复走过于相似的采样轨迹：

```json
{
  "sampler_seed_policy": "random_each_round",
  "sampler_seed_base": null
}
```

- `random_each_round`: 每轮自动生成新的 sampler seed，并写入该轮 `effective_optuna_config.json`
- `per_round_increment`: 第 1 轮用 base seed，第 2 轮用 `base+1`，依次递增
- `fixed`: 每轮都沿用当前 Optuna 配置里的 `sampler.seed`
- 每轮实际使用的配置会写到 `round_xx/effective_optuna_config.json`
- 每轮实际使用的 seed 也会记录到 `round_xx/round_meta.json`

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
