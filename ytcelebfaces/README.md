# Youtube Celebrities Faces Experiment

## Download the dataset

A page describing this dataset is located at <http://seqamlab.com/youtube-celebrities-face-tracking-and-recognition-dataset/>. Here's a [link to the dataset archive](http://seqamlab.com/wp-content/uploads/Data/ytcelebrity.tar). It is about 900 MB. Download it and unpack somewhere.

This code requires at least (approximately) 12 GB of RAM+SWAP and 5 GB of videocard memory.

## Prepare Python environment

We recommend setting up Python environment for this experiment in Linux, because we used Linux. You need python 3. There are two ways to set up the environment. After any of those you will need to install one more library.

### First way

Open <conda_requirements.txt>. The first lines of that file tell you how to quickly set up an environment suitable for reproducing this. If some packages fail to install, you might need to install them manually from pip or from other sources.

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
* ignite

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

## Running the experiment notebook

1. Open <ytcelebrities_for_public.ipynb>. Replace find `"cuda:0"` in it and, if needed, replace with your desired videocard.
2. Find assignment of `data_path` variable and change it to path where your dataset is stored. It must be the directory which contains all the video files, and there must be no other files.
3. Find assignment of `SAVE_DIR` and set it to a directory where you want the models and training history to be saved.
4. Find `HOW_FAST` variable assignment. For performing the actual experiment we did, change its value to `"medium"` as written in a comment near its assignment. If you want to quickly make sure all the code works, set it to `"fast"`.
5. Press "Run all cells" in jupyter notebook.