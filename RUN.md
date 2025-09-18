# How to Reproduce the Reported Results

Compiled in colab by running cells in Jupyter Notebook, change params as indicated in markdown. 

## 1.2 Table 1 — Expert Data (5 Environments)

Compute the **mean & std over two expert trajectories** for each environment:

```python
# Run from the hw1 directory (Python REPL or a notebook cell)
import os, pickle, numpy as np

def two_traj_stats(pkl_path):
    with open(pkl_path, 'rb') as f:
        paths = pickle.load(f)  # list of dicts with 'reward'
    rets = [float(np.sum(p['reward'])) for p in paths[:2]]  # exactly 2 trajectories
    return float(np.mean(rets)), float(np.std(rets)), rets

envs = ['Ant-v2','Humanoid-v2','Walker2d-v2','Hopper-v2','HalfCheetah-v2']
for env in envs:
    pkl = os.path.join('rob831','expert_data', f'expert_data_{env}.pkl')
    mean_ret, std_ret, rets = two_traj_stats(pkl)
    print(f'{env:12s}  mean={mean_ret:.2f}  std={std_ret:.2f}  returns={rets}')
```

Use these numbers to fill Table 1.

## 1.3 Table 2 — Behavioral Cloning (Two Tasks: Ant-v2 & Humanoid-v2)
Run BC once per task with the same architecture/data/iterations (fair comparison). These commands produced the numbers in Table 2.

## Ant-v2 (BC)
```
!python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 \
  --exp_name q1_bc_ant \
  --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --n_layers 2 --size 64 --learning_rate 5e-3 \
  --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
  --batch_size 1000 \
  --ep_len 1000 --eval_batch_size 5000 \
  --video_log_freq -1 --seed 1
```

## Humanoid-v2 (BC)
```
!python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 \
  --exp_name q1_bc_humanoid \
  --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --n_layers 2 --size 64 --learning_rate 5e-3 \
  --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
  --batch_size 1000 \
  --ep_len 1000 --eval_batch_size 5000 \
  --video_log_freq -1 --seed 1
```

From the final eval of each run, record:
	•	Eval_AverageReturn (mean over ≈5 rollouts)
	•	Eval_StdReturn (std over those rollouts)
These populate the BC row of Table 2 (the Expert row uses Table 1 means/stds). Optionally compute “% of expert” as: 100 * Eval_AverageReturn / ExpertMean.

## 1.4 Figure 1 — BC Hyperparameter Study (Vary Training Steps per Iteration)
Vary num_agent_train_steps_per_iter on Ant-v2 and keep everything else fixed.
### Replace <STEPS> with one of {100, 500, 1000, 1500, 2000}
```
!python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 \
  --exp_name bc_ant_steps<STEPS> \
  --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --n_layers 2 --size 64 --learning_rate 5e-3 \
  --num_agent_train_steps_per_iter <STEPS> --train_batch_size 100 \
  --batch_size 1000 \
  --ep_len 1000 --eval_batch_size 5000 \
  --video_log_freq -1 --seed 1
```
For each run, take the final Eval_AverageReturn (mean) and Eval_StdReturn (std) and plot mean ± std versus <STEPS>.

## 2.2 Figure 2 — DAgger Learning Curves (Two Tasks)
Run DAgger for 10 iterations on Ant-v2 and Humanoid-v2:

## Ant-v2 (DAgger)
```
!python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 \
  --exp_name q2_dagger_ant \
  --do_dagger --n_iter 10 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --n_layers 2 --size 64 --learning_rate 5e-3 \
  --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
  --batch_size 1000 \
  --ep_len 1000 --eval_batch_size 5000 \
  --video_log_freq -1 --seed 1
```
## Humanoid-v2 (DAgger)
```
!python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 \
  --exp_name q2_dagger_humanoid \
  --do_dagger --n_iter 10 \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --n_layers 2 --size 64 --learning_rate 5e-3 \
  --num_agent_train_steps_per_iter 1000 --train_batch_size 100 \
  --batch_size 1000 \
  --ep_len 1000 --eval_batch_size 5000 \
  --video_log_freq -1 --seed 1
```
Plot DAgger iteration (x-axis) vs Eval_AverageReturn (y-axis) with error bars from Eval_StdReturn.
Overlay expert and BC as horizontal baselines on each subplot (Ant on the left, the other env on the right).
