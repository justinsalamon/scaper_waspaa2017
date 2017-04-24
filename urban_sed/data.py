# CREATED: 4/6/17 10:54 by Justin Salamon <justin.salamon@nyu.edu>

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import sed_eval
import gzip
import glob

def load_urbansed_cnn(sequence_frames=50, sequence_hop=25,
                      sequence_offset=0, audio_hop=882,
                      sr=44100, verbose=True, normalize=True,
                      mel_bands=128):

    hop_time = audio_hop / float(sr)
    meta_folder = (
        '/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/metadata')
    feature_folder = ('/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/'
                      'features/logmelspec1764_{:d}/'.format(mel_bands))

    # Get label list
    label_list = (['air_conditioner', 'car_horn', 'children_playing',
                   'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
                   'jackhammer', 'siren', 'street_music'])
    label_list = sorted(label_list)
    if verbose:
        print("label list ({:d}):".format(len(label_list)))
        print(label_list)

    x_train = []
    y_train = []
    id_train = []

    x_val = []
    y_val = []
    id_val = []

    x_test = []
    y_test = []
    id_test = []

    # FIRST CONCAT ALL TRAIN TFEATURES AND FIT SCALER
    train_features = []
    train_labels = []

    # Get list of train files
    train_files = glob.glob(os.path.join(feature_folder, 'train', '*.npy.gz'))

    # Append training data
    for tf in train_files:

        label_file = os.path.join(meta_folder, 'train',
                                  os.path.basename(tf).replace('.npy.gz',
                                                               '.txt'))
        melspec = np.load(gzip.open(tf, 'rb'))
        labels = pd.read_csv(label_file, delimiter=' ', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']

        train_features.append(melspec)
        train_labels.append(labels)

    # Fit scaler OUTSIDE OF LOOP
    tf_all = []
    for tf in train_features:
        tf_all.extend(tf.T)

    tf_all = np.asarray(tf_all)
    scaler = StandardScaler()
    scaler.fit(tf_all)
    assert scaler.mean_.shape[0] == mel_bands
    assert scaler.scale_.shape[0] == mel_bands

    # THEN GENERATE ACTUAL TRAIN/VALIDATE/TEST SETS

    # Get train, validate, and test files
    train_files = glob.glob(os.path.join(feature_folder, 'train', '*.npy.gz'))
    validate_files = glob.glob(os.path.join(feature_folder, 'validate',
                                            '*.npy.gz'))
    test_files = glob.glob(os.path.join(feature_folder, 'test', '*.npy.gz'))

    # Create full training set
    for tf in train_files:

        feature_file = tf
        label_file = os.path.join(meta_folder, 'train',
                                  os.path.basename(tf).replace('.npy.gz',
                                                               '.txt'))
        melspec = np.load(gzip.open(feature_file, 'rb'))
        labels = pd.read_csv(label_file, delimiter=' ', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']

        # get y first
        event_roll = sed_eval.util.event_roll.event_list_to_event_roll(
            labels.to_dict('records'), event_label_list=label_list,
            time_resolution=hop_time)

        # Carve out x's and get aligned y's
        for i in np.arange(0 + sequence_offset,
                           melspec.shape[1] - 1 * sequence_frames,
                           sequence_hop):
            # Get x
            if normalize:
                x_train.append(
                    scaler.transform(melspec[:, i:i + sequence_frames].T).T)
            else:
                x_train.append(melspec[:, i:i + sequence_frames])
            # Get y
            y = (event_roll[i:i + sequence_frames, :]).tolist()
            if len(y) != sequence_frames:
                y.extend(np.zeros((sequence_frames - len(y), len(label_list))))
            y = np.asarray(y)
            assert y.shape == (sequence_frames, len(label_list))
            y = y.any(axis=0) * 1
            assert y.shape == (len(label_list),)
            y_train.append(y)
            # Get id
            id = [tf, i]
            id_train.append(id)

    # Create full validation set
    for tf in validate_files:

        feature_file = tf
        label_file = os.path.join(meta_folder, 'validate',
                                  os.path.basename(tf).replace('.npy.gz',
                                                               '.txt'))
        melspec = np.load(gzip.open(feature_file, 'rb'))
        labels = pd.read_csv(label_file, delimiter=' ', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']

        # get y first
        event_roll = sed_eval.util.event_roll.event_list_to_event_roll(
            labels.to_dict('records'), event_label_list=label_list,
            time_resolution=hop_time)

        # Carve out x's and get aligned y's
        for i in np.arange(0,
                           melspec.shape[1] - 1 * sequence_frames,
                           sequence_frames):
            # Get x
            if normalize:
                x_val.append(
                    scaler.transform(melspec[:, i:i + sequence_frames].T).T)
            else:
                x_val.append(melspec[:, i:i + sequence_frames])
            # Get y
            y = (event_roll[i:i + sequence_frames, :]).tolist()
            if len(y) != sequence_frames:
                y.extend(np.zeros((sequence_frames - len(y), len(label_list))))
            y = np.asarray(y)
            assert y.shape == (sequence_frames, len(label_list))
            y = y.any(axis=0) * 1
            assert y.shape == (len(label_list),)
            y_val.append(y)
            # Get id
            id = [tf, i]
            id_val.append(id)

    # Create full test set
    for tf in test_files:

        feature_file = tf
        label_file = os.path.join(meta_folder, 'test',
                                  os.path.basename(tf).replace('.npy.gz',
                                                               '.txt'))
        melspec = np.load(gzip.open(feature_file, 'rb'))
        labels = pd.read_csv(label_file, delimiter=' ', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']

        # get y first
        event_roll = sed_eval.util.event_roll.event_list_to_event_roll(
            labels.to_dict('records'), event_label_list=label_list,
            time_resolution=hop_time)

        # Carve out x's and get aligned y's
        for i in np.arange(0,
                           melspec.shape[1] - 1 * sequence_frames,
                           sequence_frames):
            # Get x
            if normalize:
                x_test.append(
                    scaler.transform(melspec[:, i:i + sequence_frames].T).T)
            else:
                x_test.append(melspec[:, i:i + sequence_frames])
            # Get y
            y = (event_roll[i:i + sequence_frames, :]).tolist()
            if len(y) != sequence_frames:
                y.extend(np.zeros((sequence_frames - len(y), len(label_list))))
            y = np.asarray(y)
            assert y.shape == (sequence_frames, len(label_list))
            y = y.any(axis=0) * 1
            assert y.shape == (len(label_list),)
            y_test.append(y)
            # Get id
            id = [tf, i]
            id_test.append(id)

    # Convert to ndarray and reshape
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    id_train = np.asarray(id_train)

    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    id_val = np.asarray(id_val)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    id_test = np.asarray(id_test)

    # Add channel dimension for Keras
    x_train = np.expand_dims(x_train, 3)
    x_val = np.expand_dims(x_val, 3)
    x_test = np.expand_dims(x_test, 3)

    if verbose:
        print('Data shapes:')
        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)

    # print(shapesum)

    return (x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test,
            id_test, label_list, scaler)
