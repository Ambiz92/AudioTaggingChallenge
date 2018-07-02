#! /usr/bin/env python3

from __future__ import print_function
import numpy as np
import librosa
import os
from panotti.models import *
from panotti.datautils import *
from panotti.multi_gpu import MultiGPUModelCheckpoint
from os.path import isfile
from timeit import default_timer as timer


def train_network(weights_file, classpath, epochs, batch_size, val_split, tile=False):
    # Instantiate a random seed for reproducible results
    np.random.seed(1)

    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path = classpath, batch_size = batch_size, tile = tile)

    # Instantiate the model
    model, serial_model = setup_model(X_train, class_names, weights_file = weights_file)

    save_best_only = (val_split > 1e-6)
    checkpointer = MultiGPUModelCheckpoint(filepath = weights_file, verbose = 1, save_best_only = save_best_only, serial_model = serial_model, period = 1, class_names = class_names)

    # Train the network
    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, shuffle = True, verbose = 1, callbacks = [checkpointer], validation_split = val_split)

    # Score the model against test dataset
    X_test, Y_test, paths_test, class_names_test = build_dataset(path = classpath + '../Test/', tile = tile)
    assert(class_names == class_names_test)
    # Evaluate the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    import argparse
    # Add list of command line arguments
    parser = argparse.ArgumentParser(description = 'Neural network trainer')
    parser.add_argument('-w', '--weights', help = 'Weights file in hdf5 format', default = 'weights.hdf5')
    parser.add_argument('-c', '--classpath', help = 'Training dataset directory with list of classes', default = 'Preproc/Train/')
    parser.add_argument('--epochs', default = 20, type = int, help = 'Number of iterations to train for')
    parser.add_argument('--batch_size', default = 40, type = int, help = 'Number of clips to send to GPU at once')
    parser.add_argument('--val', default = 0.25, type = float, help = 'Fraction of training set to split off for validation')
    parser.add_argument("--tile", help = 'Tile mono spectrograms 3 times for use with imagenet models', action = 'store_true')
    # Parse the command line arguments
    args = parser.parse_args()

    # Train the network with specified parameters
    train_network(weights_file = args.weights, classpath = args.classpath, epochs = args.epochs, batch_size = args.batch_size, val_split = args.val, tile = args.tile)
