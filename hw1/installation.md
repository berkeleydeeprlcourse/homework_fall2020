## Install mujoco:
```
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
rm mujoco200_linux.zip
cp <location_of_mjkey.txt> .
```
The above instructions download MuJoCo for Linux. If you are on Mac or Windows, you will need to change the `wget` address to either 
`https://www.roboti.us/download/mujoco200_macos.zip` or `https://www.roboti.us/download/mujoco200_win64.zip`.

Finally, add the following to bottom of your bashrc:
```
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin/
```

## Install other dependencies


There are two options:

A. (Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	2. Create a conda environment that will contain python 3:
	```
	conda create -n cs285 python=3.6
	```

	3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	source activate cs285
	```

	4. Install the requirements into this conda environment
	```
	pip install --user -r requirements.txt
	```

	5. Allow your code to be able to see 'cs285'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python:
	```
	pip install -r requirements.txt
	```