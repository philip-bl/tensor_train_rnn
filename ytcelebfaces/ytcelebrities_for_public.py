#!/usr/bin/env python
# coding: utf-8

# In[1]:

import logging
import sys
import datetime
import time
import os
import pickle
import random
from functools import partial, reduce
import operator

import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

from PIL import Image

from tqdm import tqdm

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torch.utils.data import DataLoader, TensorDataset, Dataset

# need patched version
from t3nsor import TTLinear

import click
import click_log

from ignite.engine import (
    create_supervised_evaluator, create_supervised_trainer,
    Events
)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint
import functools
import pickle

from libcrap.torch.click import click_seed_and_device_options, click_dataset_root_option


logger = logging.getLogger()
click_log.basic_config(logger)


def load_video(filename, resizing_size):
    resized_frames = (
        np.array(Image.fromarray(frame).resize(resizing_size, resample=Image.BICUBIC))
        for frame in imageio.mimread(filename, "ffmpeg")
    )
    
    return np.array(list(resized_frames))


def permute_components(video: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    num_frames, frame_len = video.shape
    assert permutation.shape == (frame_len, )
    return video[:, permutation]


def load_data_v2(full_paths, resizing_size, maxlen, shuffle_pixels):
    X = np.zeros((len(full_paths), maxlen, np.product(resizing_size) * 3), dtype=np.int8)
    if shuffle_pixels:
        permutation = np.arange(np.prod(resizing_size) * 3)
        np.random.shuffle(permutation)
    for (n, path) in enumerate(tqdm(full_paths)):
        this_clip = load_video(path, resizing_size)
        this_clip = this_clip.reshape(this_clip.shape[0], -1).astype(np.int8)
        this_clip = (this_clip - 128)
        if shuffle_pixels:
            this_clip = permute_components(this_clip, permutation)
        X[n] = pad_sequences([this_clip], maxlen=maxlen, truncating="post", dtype=np.int8)[0]
    return X


class GRU(nn.Module):
    def __init__(
        self, in_features, out_features, sequence_length, hidden_state_length,
        linear_layer_maker, dropout_prob, updated_hidden_state_activation_maker=torch.nn.Tanh
    ):
        super().__init__()
        self.hidden_state_length = hidden_state_length
        self.sequence_length = sequence_length
        
        #initialize components required for calculating $ d^{(t)} $
        #i.e. updated hidden value, which gets elementwisely multiplied by (1 - u{(t-1)})
        # where u is update gate
        # remark: hidden state is h^{(t)},
        # updated hidden value is d^{(t)}
        self.input_to_updated_hidden_value = linear_layer_maker(
            in_features, hidden_state_length, bias=True
        )
        self.hidden_state_to_updated_hidden_value = nn.Linear(
            hidden_state_length, hidden_state_length, bias=False
        )
        self.reset_gate = None
        self.updated_hidden_value_dropout = nn.Dropout(p=dropout_prob)
        self.updated_hidden_state_activation = updated_hidden_state_activation_maker()
        
        # initialize components required for calculating $ h^{(t)}, i.e.
        # hidden state
        # here u^{(t-1)} is update gate which, if takes value one,
        # means that new value of h should just equal prev value of h`
        self.hidden_state = None
        self.update_gate = None
        
        # initialize components for calculating update gate
        self.input_to_update_gate = linear_layer_maker(
            in_features, hidden_state_length, bias=True
        )
        self.hidden_state_to_update_gate = nn.Linear(
            hidden_state_length, hidden_state_length, bias=False
        )
        
        # initialize components for calculating reset gate
        self.input_to_reset_gate = linear_layer_maker(
            in_features, hidden_state_length, bias=True
        )
        self.hidden_state_to_reset_gate = nn.Linear(
            hidden_state_length, hidden_state_length, bias=False
        )
        
        # initialize the classifier which performs classification
        # of last hidden state
        self.last_state_to_output = nn.Linear(hidden_state_length, out_features, bias=True)
    
    def forward(self, X):
        assert len(X.shape) == 3
        num_samples = X.shape[0]
        assert X.shape[1] == self.sequence_length
        
        self.hidden_state, self.reset_gate, self.update_gate = [
            torch.zeros(num_samples, self.hidden_state_length, device=X.device)
            for i in range(3)
        ]
        
        for part_index in range(self.sequence_length):
            X_part = X[:, part_index].contiguous()
            
            # this is also called d^{(t)}
            updated_hidden_state_value = self.updated_hidden_value_dropout(self.updated_hidden_state_activation(
                self.input_to_updated_hidden_value(X_part)
                + self.hidden_state_to_updated_hidden_value(self.hidden_state * self.reset_gate)
            ))
            
            # calculate hidden state h^{(t)}
            self.hidden_state = self.update_gate * self.hidden_state                 + (1 - self.update_gate) * updated_hidden_state_value
            
            # calculate update gate u^{(t)} which will affect the next iteration
            self.update_gate = torch.sigmoid(
                self.input_to_update_gate(X_part)
                + self.hidden_state_to_update_gate(self.hidden_state)
            )
            
            # calculate reset gate r^{(t)} which will affect the next iteration
            self.reset_gate = torch.sigmoid(
                self.input_to_reset_gate(X_part)
                + self.hidden_state_to_reset_gate(self.hidden_state)
            )
        
        # we have calculated the last hidden state
        # now we calculate logits
        return self.last_state_to_output(self.hidden_state)


class BuiltinGRU(nn.Module):
    def __init__(self, in_features, out_features, hidden_state_length):
        super().__init__()
        self.hidden_state_length = hidden_state_length
        self.gru = nn.GRU(in_features, hidden_state_length, batch_first=True)
        self.linear = nn.Linear(hidden_state_length, out_features)

    def forward(self, X):
        gru_last_hidden = self.gru(X)[0][:, -1]
        return self.linear(gru_last_hidden)


class NumpyArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays
    
    def __len__(self):
        return len(self.arrays[0])
    
    def __getitem__(self, idx):
        return tuple(
            torch.from_numpy(array[idx].astype(np.float32)) if len(array[idx].shape) != 0 else torch.tensor(array[idx])
            for array in self.arrays
        )


class TrainingHistory(object):
    """History of how {train,test} {loss,accuracy} changes as training goes."""
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
    
    def show_plots(self, output_dir):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
        axes = axes.flatten()
        
        axes[0].set_title("Loss")
        axes[0].plot(self.epochs, self.train_losses, label="train loss")
        axes[0].plot(self.epochs, self.validation_losses, label="validation loss")
        axes[0].legend()
        
        axes[1].set_title(
            "Accuracy. Last val: {}, best val: {}".format(
                self.validation_accuracies[-1],
                max(self.validation_accuracies)
            )
        )
        axes[1].plot(self.epochs, self.train_accuracies, label="train accuracy")
        axes[1].plot(
            self.epochs, self.validation_accuracies,
            label="validation accuracy"
        )
        axes[1].legend()
        fig.savefig(os.path.join(output_dir, "plots.png"))


def train_and_evaluate(
    model, optimizer,
    train_loader, validation_loader,
    eval_every_num_epochs, plot_every_num_epochs,
    num_epochs, early_stopping_epochs,
    save_dir, save_prefix, criterion, device,
    output_dir
):
    """The main neural network training function. Trains a nn,
    evaluates it as it goes. Saves the best model in save_dir.
    The filename will have prefix save_prefix. Performs training
    for at most num_epochs. If validation accuracy doesn't improve for
    early_stopping_epochs, stops training process. Plots training
    history during training.
    
    eval_every_num_epochs - how often to evaluate.
    
    plot_every_num_epochs - how often to update plots."""
    trainer = create_supervised_trainer(
        model=model, optimizer=optimizer,
        loss_fn=criterion, device=device
    )
    
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "accuracy": Accuracy(),
            "loss": Loss(criterion)
        },
        device=device
    )
    
    history = TrainingHistory()
    
    def evaluate(loader, loss_log, accuracy_log):
        model.train(False)
        evaluator.run(loader)
        loss_log.append(evaluator.state.metrics["loss"])
        accuracy_log.append(evaluator.state.metrics["accuracy"])

    @trainer.on(Events.EPOCH_COMPLETED)
    @do_every_num_epochs(eval_every_num_epochs)
    def evaluate_on_train_and_test(engine):
        evaluate(train_loader, history.train_losses, history.train_accuracies)
        evaluate(validation_loader, history.validation_losses, history.validation_accuracies)
        assert not isinstance(engine.state.epoch, torch.Tensor)
        history.epochs.append(engine.state.epoch)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    @do_every_num_epochs(plot_every_num_epochs)
    def update_plot(*args):
        history.show_plots(output_dir)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, EarlyStopping(
        patience=early_stopping_epochs, # wait this many epochs before stopping
        score_function=lambda engine: history.validation_accuracies[-1],
        trainer=trainer
    ))
    
    # Add handler which saves model to disk whenever it achieves
    # new best result
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            save_dir, save_prefix,
            score_function=lambda engine: history.validation_accuracies[-1],
            n_saved=1, atomic=True, require_empty=False,
            save_as_state_dict=True
        ),
        {"model": model} # what should be saved
    )
        
    trainer.run(train_loader, max_epochs=num_epochs)
    
    # save training history to disk as well
    with open(os.path.join(save_dir, f"{save_prefix}_history.pkl"), "wb") as history_f:
        pickle.dump(history, history_f)
    
    # usually we will not be using return values, because it returns
    # last model, not the best model
    return (
        model,
        history
    )


def do_every_num_epochs(num_epochs):
    """This must be written after @trainer.on, not before. Run function
    every few epochs instead of every epoch."""
    def decorate(func):
        def decorated(engine, *args, **kwargs):
            if engine.state.epoch % num_epochs == 0:
                return func(engine, *args, **kwargs)
        return functools.update_wrapper(decorated, func)
    return decorate


@click.command()
@click_dataset_root_option()
@click.option(
    "--output-dir", "-o", required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True),
    help="Dir where checkpoint(s) of models and plots will be saved."
)
@click.option("--shuffle-components/--no-shuffle-components", default=False)
@click.option("--how-fast", type=click.Choice(("slow", "medium", "fast")), default="medium")
@click.option("--long-seq/--no-long-seq", default=False)
@click.option("--num-epochs", type=int, required=True)
@click.option("--learning-rate", type=float, required=True)
@click_log.simple_verbosity_option(logger)
@click_seed_and_device_options()
def main(
    seed: int, device, shuffle_components: bool, how_fast: str, long_seq: False,
    dataset_root: str, output_dir: str, num_epochs: int, learning_rate: float
):
    resizing_size = {
        "slow": (160, 120),
        "medium": (80, 60),
        "fast": (29, 13)
    }[how_fast]
    global_max_len = 350 if long_seq else 85

    files = os.listdir(dataset_root)
    random.shuffle(files)
    targets = [''] * len(files)
    for l in range(len(files)):
        # l = 0
        this_file = files[l]

        # files are named like 
        this_file_split = this_file.split('_')

        # remove .avi or .whatever extension
        this_file_split[-1] = this_file_split[-1].split('.')[0]

        # set targets (labels) to strings like "vladimir_putin"
        targets[l] = this_file_split[-2] + '_' + this_file_split[-1]

    targets = np.array(targets)

    # np.array of unique target labels
    classes = np.unique(targets)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(targets)
    NUM_TRAIN, NUM_TEST = {
        "fast": (32, 16),
        "medium": (len(files) - 200, 200),
        "slow": (len(files) - 200, 200)
    }[how_fast]

    N = NUM_TRAIN + NUM_TEST

    files = files[:N]
    targets = targets[:N]
    y = y[:N]
    full_paths = np.array([os.path.join(dataset_root, this_file) for this_file in files])
    X = load_data_v2(full_paths, resizing_size=resizing_size, maxlen=global_max_len, shuffle_pixels=shuffle_components)
    np.savez(os.path.join(output_dir, "Xy.npz"), X, y)
    X_train = X[:NUM_TRAIN]
    X_test = X[NUM_TRAIN:]

    y_train = y[:NUM_TRAIN]
    y_test = y[NUM_TRAIN:]

    full_paths_train = full_paths[:NUM_TRAIN]
    full_paths_test = full_paths[NUM_TRAIN:]
    num_features = reduce(operator.mul, resizing_size) * 3
    num_labels = len(classes)
    dataset_train = NumpyArrayDataset(X_train, y_train)
    dataset_test = NumpyArrayDataset(X_test, y_test)
    BATCH_SIZE = {"fast": 32, "medium": 64, "slow": 8}[how_fast]
    train_loader = DataLoader(
        dataset_train, batch_size=min(BATCH_SIZE, NUM_TRAIN), shuffle=True, drop_last=True,
        num_workers=1, pin_memory=True
    )
    val_loader = DataLoader(
        dataset_test, batch_size=min(BATCH_SIZE, NUM_TEST), shuffle=True, drop_last=False,
        num_workers=1, pin_memory=True
    )
    dropout_prob = 0.4
    tt_rank = 3
    model = GRU(
        num_features, num_labels,
        sequence_length=global_max_len,
        hidden_state_length=256,
        linear_layer_maker=partial(TTLinear, d=4, tt_rank=tt_rank),
        dropout_prob=dropout_prob,
        updated_hidden_state_activation_maker=torch.nn.ELU
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, weight_decay=1e-2, amsgrad=True
    )
    criterion = tnnf.cross_entropy
    train_and_evaluate(
        model,
        optimizer,
        train_loader,
        val_loader,
        eval_every_num_epochs=1,
        plot_every_num_epochs=1,
        num_epochs=num_epochs,
        early_stopping_epochs=100,
        save_dir=output_dir,
        save_prefix=f"tt_gru_activation_elu_adam_seed_{seed}_{how_fast}_dropout_prob_{dropout_prob}_rank_{tt_rank}_lr={learning_rate:.1E}_shuffle={shuffle_components}",
        criterion=criterion, device=device, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
