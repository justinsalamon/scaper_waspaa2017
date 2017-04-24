# CREATED: 4/6/17 11:05 by Justin Salamon <justin.salamon@nyu.edu>

from scaper_waspaa2017.urban_sed.data import load_urbansed_cnn
from scaper_waspaa2017.urban_sed.util import event_roll_to_event_list
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers.core import Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras
import tensorflow
import sklearn
import numpy as np
import os
import gzip
import json
import time
import sed_eval
import pickle


def build_custom_cnn(n_freq_cnn=128, n_frames_cnn=50, n_filters_cnn=64,
                     filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                     n_classes=10, large_cnn=False, n_dense_cnn=64):

    if large_cnn:
        n_filters_cnn = 128
        n_dense_cnn = 128

    # INPUT
    x = Input(shape=(n_freq_cnn, n_frames_cnn, 1), dtype='float32')

    # CONV 1
    y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(x)
    y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # CONV 2
    y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # CONV 3
    y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    # y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # Flatten for dense layers
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(n_dense_cnn, activation='relu')(y)
    if large_cnn:
        y = Dropout(0.5)(y)
        y = Dense(n_dense_cnn, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(n_classes, activation='sigmoid')(y)

    m = Model(inputs=x, outputs=y)
    return m


def run_experiment_sedeval(expid, epochs=1000, metrics=['accuracy'],
                           sequence_hop=25,
                           batch_size=64, audio_hop=882, sr=44100,
                           sequence_frames=50,
                           epoch_limit=2048,
                           sed_early_stopping=100, normalize_data=True,
                           fit_verbose=True, mel_bands=40, resume=False,
                           resume_f1_best=0, load_subset=None,
                           large_cnn=False):
    # Print out library versions
    print('Module versions:')
    print('keras version: {:s}'.format(keras.__version__))
    print('tensorflow version: {:s}'.format(tensorflow.__version__))
    print('numpy version: {:s}'.format(np.__version__))
    print('sklearn version: {:s}'.format(sklearn.__version__))

    # Create output folders
    modelfolder = '/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/models'
    expfolder = os.path.join(modelfolder, expid)
    time.sleep(np.random.rand() * 10)  # prevents jobs clashing
    if not os.path.isdir(expfolder):
        os.mkdir(expfolder)

    # Load data
    (x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test,
     id_test, label_list, scaler) = load_urbansed_cnn(
        sequence_frames=sequence_frames,
        sequence_hop=sequence_hop,
        sequence_offset=0,
        audio_hop=audio_hop, sr=sr,
        verbose=True,
        normalize=normalize_data,
        mel_bands=mel_bands,
        load_subset=load_subset)

    # Save scaler object
    scaler_file = os.path.join(expfolder, 'scaler.pkl')
    pickle.dump(scaler, open(scaler_file, 'wb'))

    # Build model
    print('\nBuilding model...')
    model = build_custom_cnn(n_freq_cnn=mel_bands, large_cnn=large_cnn)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=metrics)

    if resume:
        print('Loading best weights and resuming...')
        weights_best_file = os.path.join(expfolder, 'weights_best.hdf5')
        model.load_weights(weights_best_file)

    # Fit model
    print('\nFitting model...')
    history_sed = []
    f1s_best = 0
    epoch_best = 0
    epochs_since_improvement = 0

    if resume:
        f1s_best = resume_f1_best

    for epoch in range(epochs):

        if epoch_limit is None:

            history = model.fit(x=x_train, y=y_train, batch_size=batch_size,
                                epochs=epoch+1, verbose=fit_verbose,
                                validation_split=0.0,
                                validation_data=(x_val, y_val), shuffle=True,
                                initial_epoch=epoch)

            # Test using sed_eval
            pred = model.predict(x_val)
            # est_roll, ref_roll = combine_event_rolls(pred, y_test, threshold=0.5)
            ref_roll = y_val[:]
            est_roll = 1 * (pred >= 0.5)
            est_event_list = event_roll_to_event_list(
                est_roll, label_list, 1.0)
            ref_event_list = event_roll_to_event_list(
                ref_roll, label_list, 1.0)

            seg_metrics1s = sed_eval.sound_event.SegmentBasedMetrics(
                label_list, time_resolution=1.0)
            seg_metrics1s.evaluate(ref_event_list, est_event_list)
            results1s = seg_metrics1s.results()

            seg_metrics100ms = sed_eval.sound_event.SegmentBasedMetrics(
                label_list, time_resolution=0.1)
            seg_metrics100ms.evaluate(ref_event_list, est_event_list)
            results100ms = seg_metrics100ms.results()

            scores = (
                'F1s: {:.4f}, P1s: {:.4f}, R1s: {:.4f}, F100ms: {:.4f}, '
                'P100ms: {:.4f}, R100ms: {:.4f} | E1s: {:.4f}, '
                'E100ms: {:.4f}'.format(
                    results1s['overall']['f_measure']['f_measure'],
                    results1s['overall']['f_measure']['precision'],
                    results1s['overall']['f_measure']['recall'],
                    results100ms['overall']['f_measure']['f_measure'],
                    results100ms['overall']['f_measure']['precision'],
                    results100ms['overall']['f_measure']['recall'],
                    results1s['overall']['error_rate']['error_rate'],
                    results100ms['overall']['error_rate']['error_rate']
                ))
            history_sed.append([results1s, results100ms])
            print(scores)

            # Save weights if we see improvement
            f1s_current = results1s['overall']['f_measure']['f_measure']
            improvement = False
            if f1s_current > f1s_best:
                improvement = True
                f1s_best = f1s_current
                weights_best_file = os.path.join(
                    expfolder, 'weights_best.hdf5')
                model.save_weights(weights_best_file)

                # ********************* CHECKPOINT ************************
                # Save history.history
                history_score_file = os.path.join(expfolder,
                                                  'history_scores.json')
                json.dump(history.history, open(history_score_file, 'w'),
                          indent=2)
                # Save history_sed
                history_sed_file = os.path.join(expfolder, 'history_sed.json')
                json.dump(history_sed, open(history_sed_file, 'w'), indent=2)
                # Get predictions
                pred = model.predict(x_test)
                # Save Y_test, predictions and IDs
                ytestfile = os.path.join(expfolder, 'ytest.npy.gz')
                yprobfile = os.path.join(expfolder, 'yprob.npy.gz')
                yidfile = os.path.join(expfolder, 'yid.npy.gz')
                y_test.dump(gzip.open(ytestfile, 'wb'))
                pred.dump(gzip.open(yprobfile, 'wb'))
                id_test.dump(gzip.open(yidfile, 'wb'))

            if improvement:
                print('{:d} Best val F1s: {:.4f} (IMPROVEMENT, saving)\n'.format(
                    epoch, f1s_best))
                epochs_since_improvement = 0
                epoch_best = epoch
            else:
                print('{:d} Best val F1s: {:.4f} ({:d})\n'.format(
                    epoch, f1s_best, epoch_best))
                epochs_since_improvement += 1

        else:

            order = np.arange(x_train.shape[0])
            np.random.shuffle(order)

            idx = 0
            mini_epochs_per_epoch = int(
                np.floor((x_train.shape[0] - epoch_limit)/
                         float(epoch_limit))) + 1
            mini_epoch = epoch * mini_epochs_per_epoch
            while idx < x_train.shape[0] - epoch_limit:

                history = model.fit(x=x_train[order[idx:idx+epoch_limit]],
                                    y=y_train[order[idx:idx+epoch_limit]],
                                    batch_size=batch_size,
                                    epochs=mini_epoch+1, verbose=fit_verbose,
                                    validation_split=0.0,
                                    validation_data=(x_val, y_val), shuffle=True,
                                    initial_epoch=mini_epoch)

                idx += epoch_limit
                mini_epoch += 1

                # Test using sed_eval
                pred = model.predict(x_val)
                # est_roll, ref_roll = combine_event_rolls(pred, y_test,
                #                                          threshold=0.5)
                ref_roll = y_val[:]
                est_roll = 1 * (pred >= 0.5)
                est_event_list = event_roll_to_event_list(
                    est_roll, label_list, 1.0)
                ref_event_list = event_roll_to_event_list(
                    ref_roll, label_list, 1.0)

                seg_metrics1s = sed_eval.sound_event.SegmentBasedMetrics(
                    label_list, time_resolution=1.0)
                seg_metrics1s.evaluate(ref_event_list, est_event_list)
                results1s = seg_metrics1s.results()

                seg_metrics100ms = sed_eval.sound_event.SegmentBasedMetrics(
                    label_list, time_resolution=0.1)
                seg_metrics100ms.evaluate(ref_event_list, est_event_list)
                results100ms = seg_metrics100ms.results()

                scores = (
                    'F1s: {:.4f}, P1s: {:.4f}, R1s: {:.4f}, F100ms: {:.4f}, '
                    'P100ms: {:.4f}, R100ms: {:.4f} | E1s: {:.4f}, '
                    'E100ms: {:.4f}'.format(
                        results1s['overall']['f_measure']['f_measure'],
                        results1s['overall']['f_measure']['precision'],
                        results1s['overall']['f_measure']['recall'],
                        results100ms['overall']['f_measure']['f_measure'],
                        results100ms['overall']['f_measure']['precision'],
                        results100ms['overall']['f_measure']['recall'],
                        results1s['overall']['error_rate']['error_rate'],
                        results100ms['overall']['error_rate']['error_rate']
                    ))
                history_sed.append([results1s, results100ms])
                print(scores)

                # Save weights if we see improvement
                f1s_current = results1s['overall']['f_measure']['f_measure']
                improvement = False
                if f1s_current > f1s_best:
                    improvement = True
                    f1s_best = f1s_current
                    weights_best_file = os.path.join(
                        expfolder, 'weights_best.hdf5')
                    model.save_weights(weights_best_file)

                    # ********************* CHECKPOINT **********************
                    # Save history.history
                    history_score_file = os.path.join(expfolder,
                                                      'history_scores.json')
                    json.dump(history.history, open(history_score_file, 'w'),
                              indent=2)
                    # Save history_sed
                    history_sed_file = os.path.join(expfolder,
                                                    'history_sed.json')
                    json.dump(history_sed, open(history_sed_file, 'w'),
                              indent=2)
                    # Get predictions
                    pred = model.predict(x_test)
                    # Save Y_test, predictions and IDs
                    ytestfile = os.path.join(expfolder, 'ytest.npy.gz')
                    yprobfile = os.path.join(expfolder, 'yprob.npy.gz')
                    yidfile = os.path.join(expfolder, 'yid.npy.gz')
                    y_test.dump(gzip.open(ytestfile, 'wb'))
                    pred.dump(gzip.open(yprobfile, 'wb'))
                    id_test.dump(gzip.open(yidfile, 'wb'))

                if improvement:
                    print('{:d} Best val F1s: {:.4f} (IMPROVEMENT, '
                          'saving)\n'.format(mini_epoch, f1s_best))
                    epochs_since_improvement = 0
                    epoch_best = mini_epoch
                else:
                    print('{:d} Best val F1s: {:.4f} ({:d})\n'.format(
                        mini_epoch, f1s_best, epoch_best))
                    epochs_since_improvement += 1

                if ((sed_early_stopping is not None) and
                        (epochs_since_improvement >= sed_early_stopping)):
                    print('\nNo improvement for {:d} epochs, stopping.'.format(
                        sed_early_stopping))
                    break

        if ((sed_early_stopping is not None) and
                (epochs_since_improvement >= sed_early_stopping)):
            print('\nNo improvement for {:d} epochs, stopping.'.format(
                sed_early_stopping))
            break

    # # Training curves
    # plot_training_curves(history)

    # Save model and predictions to disk
    print('\nSaving model and predictions to disk...')
    modeljsonfile = os.path.join(expfolder, 'model.json')
    model_json = model.to_json()
    with open(modeljsonfile, 'w') as json_file:
        json.dump(model_json, json_file, indent=2)

    # Save last version of weights (for resuming training)
    weights_last_file = os.path.join(
        expfolder, 'weights_last.hdf5')
    model.save_weights(weights_last_file)

    # Save history.history
    history_score_file = os.path.join(expfolder, 'history_scores.json')
    json.dump(history.history, open(history_score_file, 'w'), indent=2)

    # Save history_sed
    history_sed_file = os.path.join(expfolder, 'history_sed.json')
    json.dump(history_sed, open(history_sed_file, 'w'), indent=2)

    # Save scaler object
    scaler_file = os.path.join(expfolder, 'scaler.pkl')
    pickle.dump(scaler, open(scaler_file, 'wb'))

    # Save Y_test, predictions and IDs
    ytestfile = os.path.join(expfolder, 'ytest.npy.gz')
    yprobfile = os.path.join(expfolder, 'yprob.npy.gz')
    yidfile = os.path.join(expfolder, 'yid.npy.gz')

    y_test.dump(gzip.open(ytestfile, 'wb'))
    id_test.dump(gzip.open(yidfile, 'wb'))

    # Get predictions
    model.load_weights(weights_best_file)
    pred = model.predict(x_test)
    pred.dump(gzip.open(yprobfile, 'wb'))

    print("Done.")
