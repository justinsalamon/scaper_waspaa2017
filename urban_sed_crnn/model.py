# CREATED: 4/6/17 11:05 by Justin Salamon <justin.salamon@nyu.edu>

from scaper_waspaa2017.urban_sed_crnn.data import load_urbansed_crnn
from scaper_waspaa2017.urban_sed.util import event_roll_to_event_list
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, GRU
from keras.layers import Dense, Flatten, Activation, TimeDistributed
from keras.layers.core import Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras.backend as K
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


def makeCNNCell(inputHeight):
    # inputHeight: number of frequency bins

    reshape = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 5 * 96, -1, 1)))
    squeeze = Lambda(lambda x: K.squeeze(x, -1))
    transpose = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))

    x = Input(shape=(inputHeight, None, 1), dtype='float32')

    y = Conv2D(96, (5, 5), padding='same', activation=None)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid')(y)

    y = Conv2D(96, (5, 5), padding='same', activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid')(y)

    y = Conv2D(96, (5, 5), padding='same', activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid')(y)

    # reshape input for loading into a RNN
    y = reshape(y)
    y = squeeze(y)
    y = transpose(y)

    m = Model(inputs=x, outputs=y)

    return m


def makeRNNCell(inputHeight, numFreq, rnnSize, f=GRU):
    # inputHeight: dim of input
    # numFreq: dim of ouput

    # input
    x = Input(shape=(None, inputHeight), dtype='float32')

    y = f(96, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)(x)
    y = f(96, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)(y)
    y = f(96, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)(y)
    y = TimeDistributed(Dense(17, activation='sigmoid'))(y)

    m = Model(inputs=x, outputs=y)
    return m


def build_crnn(n_freq_cnn=40, n_freq_rnn=10, input_height_rnn=480, rnn_size=96,
               rnn_f=GRU):

    c = makeCNNCell(n_freq_cnn)
    r = makeRNNCell(input_height_rnn, n_freq_rnn, rnn_size, f=rnn_f)

    model_input = Input(shape=(n_freq_cnn, None, 1), dtype='float32',
                        name='model_input')
    x = c(model_input)
    out = r(x)

    model = Model(inputs=model_input, outputs=out)

    return model


def run_experiment_sedeval(expid, epochs=1000, metrics=['accuracy'],
                           sequence_hop=64,
                           batch_size=64, audio_hop=882, sr=44100,
                           sequence_frames=128,
                           epoch_limit=2048,
                           sed_early_stopping=100, normalize_data=True,
                           fit_verbose=True, mel_bands=40, resume=False,
                           resume_f1_best=0, load_subset=None):
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
     id_test, label_list, scaler) = load_urbansed_crnn(
        sequence_frames=sequence_frames,
        sequence_hop=sequence_hop,
        sequence_offset=0,
        audio_hop=audio_hop, sr=sr,
        verbose=True,
        normalize=normalize_data,
        mel_bands=mel_bands,
        load_subset=load_subset)

    # Build model
    print('\nBuilding model...')
    model = build_crnn(n_freq_cnn=mel_bands)
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
