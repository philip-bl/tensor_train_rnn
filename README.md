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
