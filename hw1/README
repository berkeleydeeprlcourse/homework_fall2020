
1) install package by running:

$ python setup.py develop

##############################################
##############################################

2)install mujoco:
$ cd ~
$ mkdir .mujoco
$ cd <location_of_your_mjkey.txt>
$ cp mjkey.txt ~/.mujoco/
$ cd <this_repo>/downloads
$ cp -r mjpro150 ~/.mujoco/

add the following to bottom of your bashrc:
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/

NOTE IF YOU'RE USING A MAC:
The provided mjpro150 folder is for Linux. 
Please download the OSX version yourself, from https://www.roboti.us/index.html

##############################################
##############################################

3)install other dependencies

-------------------

a) [PREFERRED] Option A:

i) install anaconda, if you don't already have it:
Download Anaconda2 (suggested v5.2 for linux): https://www.continuum.io/downloads
$ cd Downloads
$ bash Anaconda2-5.2.0-Linux-x86_64.sh #file name might be slightly different, but follows this format

Note that this install will modify the PATH variable in your bashrc.
You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

ii) create a conda env that will contain python 3:
$ conda create -n cs285_env python=3.5

iii) activate the environment (do this every time you open a new terminal and want to run code):
$ source activate cs285_env

iv) install the requirements into this conda env
$ pip install --user --requirement requirements.txt

v) allow your code to be able to see 'cs285'
$ cd <path_to_hw>
$ pip install -e .

Note: This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.

-------------------

b) Option B:

install dependencies locally, by running:
$ pip install -r requirements.txt

##############################################
##############################################

4) code:

Blanks to be filled in are marked with "TODO"
The following files have blanks in them:
- scripts/run_hw1_behavior_cloning.py
- infrastructure/rl_trainer.py
- agents/bc_agent.py
- policies/MLP_policy.py
- infrastructure/replay_buffer.py
- infrastructure/utils.py
- infrastructure/tf_utils.py

See the code + the hw pdf for more details.

##############################################
##############################################

5) run code: 

Run the following command for Section 1 (Behavior Cloning):

$ python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl

Run the following command for Section 2 (DAGGER):
(NOTE: the --do_dagger flag, and the higher value for n_iter)

$ python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl

##############################################

6) visualize saved tensorboard event file:

$ cd cs285/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)
