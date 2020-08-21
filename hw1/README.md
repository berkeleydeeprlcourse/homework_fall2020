## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoco and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2020/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. You will be focusing most of your attention on the following files:
1. [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)
2. [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
3. [agents/bc_agent.py](cs285/agents/bc_agent.py)
4. [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
5. [infrastructure/replay_buffer.py](cs285/infrastructure/replay_buffer.py)
6. [infrastructure/utils.py](cs285/infrastructure/utils.py)
7. [infrastructure/tf_utils.py](cs285/infrastructure/tf_utils.py)

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

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir cs285/data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](cs285/scripts/run_hw1.ipynb) for more details.

