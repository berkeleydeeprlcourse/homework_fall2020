## Setup

You may run this code on our own machine or on Google Colab. 

1. Run on your machine: If you choose to run locally, you will need to install MuJoco and some Python packages; see (installation.md)[installation.md] for instructions.
2. Colab: The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`
The following files have `TODO` markers in them:
- scripts/run_hw1_behavior_cloning.py
- infrastructure/rl_trainer.py
- agents/bc_agent.py
- policies/MLP_policy.py
- infrastructure/replay_buffer.py
- infrastructure/utils.py
- infrastructure/tf_utils.py

See the homework pdf for more details.

## Run the code

Run the following command for Section 1 (Behavior Cloning):

```
python cs285/scripts/run_hw1_behavior_cloning.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v2 --exp_name test_bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v2.pkl
```

Run the following command for Section 2 (DAgger):
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs285/scripts/run_hw1_behavior_cloning.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v2 --exp_name test_dagger_ant --n_iter 10 \
	--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl
```

## Visualize the saved tensorboard event file:

```
cd cs285/data/<your_log_dir>
tensorboard --logdir .
```

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)
