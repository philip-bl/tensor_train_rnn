# Tensor Train Decomposition for Recurrent Neural Networks

## Experiments

Different experiments use different libraries, so please read about each of them separately. You might even need to make different python environments for them.

### SVHN Dataset
You can find a script with the experiment in [tt_svhn.py](tt_svhn.py).
To download the dataset run [load_svhn.sh](load_svhn.sh).

You may want to change path to the dataset stored in a variable `data_path` in [tt_svhn.py](tt_svhn.py) before running the experiment.
To run the experiment simply run:
```
$ python tt_svhn.py
```

### Youtube Celebrities Faces

You can find everything related to this experiment in [ytcelebfaces](ytcelebfaces). In particular, read [ytcelebfaces/README.md](ytcelebfaces/README.md) to learn what libraries you need to install and how to download the dataset.


### Deep Shot Boundary Detection Dataset
You can simply start test procedure by calling:
```
$ sh deepsbd/test.sh
```
This command will run test script for 9 examples, located at [deepsbd/examples](deepsbd/examples).
To reproduce the full experiment you should download DeepSBD dataset(https://nsl.cs.sfu.ca/projects/DeepSBD/dataset/DeepSBD.tar.gz, more than 33GB in total) and run [deepsbd/train.py](deepsbd/train.py) with appropriate command line parameters.
