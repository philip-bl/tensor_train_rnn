# Youtube Celebrities Faces Experiment

## Prepare Python environment

We recommend setting up Python environment for this experiment in Linux, because we used Linux. You need python 3. There are two ways to set up the environment. After any of those you will need to install one more library.

### First way

Open <conda_requirements.txt>. The first lines of that file tell you how to quickly set up an environment suitable for reproducing this. If some packages fail to install, you will need to install them manually from pip or from other sources.

### The second way

Install (using your favorite package manager):
* Numpy
* Scipy
* Matplotlib
* pillow (it provides PIL module)
* tqdm
* keras (it's used only for some utility functions)
* scikit-learn (also used only for some utility functions)
* imageio and imageio-ffmpeg
* pytorch

### One more library

We use <https://github.com/KhrulkovV/tt-pytorch> implementation of Tensor Train Layer. Here's what you need to do:

```
$ git clone https://github.com/KhrulkovV/tt-pytorch.git
$ cd tt-pytorch

# make current directory state as in this commit
# so that you use exactly the same library version that we used
$ git checkout 78bbc261a7489d666a39a77e8659b177f97bae18

# that version was somewhat broken. we need to apply a patch which fixes it
# the patch is called fix_ttlayer.patch, and it's in the same directory
# as this readme. You need to put it in tt-pytorch directory
$ mv /wherever/you/saved/fix_ttlayer.patch .
$ git apply fix_ttlayer.patch  # this line applies the patch

# check that it works by looking at git diff - should contain that patch file
# and some changes in layers.py
$ git diff

# finally install the library
$ pip install .
```