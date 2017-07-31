# CREATED: 4/7/17 11:38 by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
# from scaper_waspaa2017.urban_sed_crnn.model import build_crnn_onestep
from scaper_waspaa2017.urban_sed_crnn.data import load_urbansed_crnn
from scaper_waspaa2017.urban_sed.util import event_roll_to_event_list
from scaper_waspaa2017.urban_sed.util import combine_event_rolls
import sed_eval
import os
import json
import gzip


def evaluate(expid, audio_hop=882, sr=44100, sequence_frames=128,
             sequence_hop=64, normalize_data=True, mel_bands=40,
             load_predictions=True, save_results=True):

    fold_results = []

    modelfolder = (
        '/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/models')
    expfolder = os.path.join(modelfolder, expid)

    if load_predictions:

        ytestfile = os.path.join(expfolder, 'ytest.npy.gz')
        yprobfile = os.path.join(expfolder, 'yprob.npy.gz')
        yidfile = os.path.join(expfolder, 'yid.npy.gz')
        y_test = np.load(gzip.open(ytestfile, 'rb'))
        pred_test = np.load(gzip.open(yprobfile, 'rb'))
        id_test = np.load(gzip.open(yidfile, 'rb'))

        label_list = (['air_conditioner', 'car_horn', 'children_playing',
                       'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
                       'jackhammer', 'siren', 'street_music'])
        label_list = sorted(label_list)
    else:

        # Build model and load in best weights
        model = build_crnn_onestep(n_freq_cnn=mel_bands)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        weightfile = os.path.join(expfolder, 'weights_best.hdf5'.format(expid))
        model.load_weights(weightfile)

        # Load data
        (x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test,
         id_test, label_list, scaler) = load_urbansed_crnn(
            sequence_frames=sequence_frames, sequence_hop=sequence_hop,
            sequence_offset=0,
            audio_hop=audio_hop, sr=sr,
            verbose=False, normalize=normalize_data,
            mel_bands=mel_bands)


        pred_test = model.predict(x_test)


    est_roll, ref_roll = combine_event_rolls(pred_test, y_test,
                                             threshold=0.5)
    # ref_roll = y_test[:]
    # est_roll = 1 * (pred_test >= 0.5)

    # COMPUTE FOLD RESULTS AND REPORT
    # Convert event rolls into even lists
    est_event_list = event_roll_to_event_list(
        est_roll, label_list, audio_hop/float(sr))
    ref_event_list = event_roll_to_event_list(
        ref_roll, label_list, audio_hop/float(sr))

    # Compute metrics at 1s, 100ms and 20ms levels
    seg_metrics1s = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=1.0)
    seg_metrics1s.evaluate(ref_event_list, est_event_list)
    results1s = seg_metrics1s.results()

    seg_metrics100ms = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=0.1)
    seg_metrics100ms.evaluate(ref_event_list, est_event_list)
    results100ms = seg_metrics100ms.results()

    seg_metrics20ms = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=0.020)
    seg_metrics20ms.evaluate(ref_event_list, est_event_list)
    results20ms = seg_metrics20ms.results()

    fold_results.append([results1s, results100ms, results20ms])

    # Report scores
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
    print(scores)

    # Save scores
    results_all = {'results1s': results1s,
                   'results100ms': results100ms,
                   'results20ms': results20ms}

    if save_results:
        results_all_file = os.path.join(expfolder, 'test_results.json')
        json.dump(results_all, open(results_all_file, 'w'), indent=2)

    return results_all


def evaluate_file_list(expid, file_list, audio_hop=882, sr=44100,
                       sequence_frames=128,
                       sequence_hop=64, normalize_data=True, mel_bands=40,
                       load_predictions=True):

    fold_results = []

    modelfolder = (
        '/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/models')
    expfolder = os.path.join(modelfolder, expid)

    if load_predictions:

        ytestfile = os.path.join(expfolder, 'ytest.npy.gz')
        yprobfile = os.path.join(expfolder, 'yprob.npy.gz')
        yidfile = os.path.join(expfolder, 'yid.npy.gz')
        y_test = np.load(gzip.open(ytestfile, 'rb'))
        pred_test = np.load(gzip.open(yprobfile, 'rb'))
        id_test = np.load(gzip.open(yidfile, 'rb'))

        label_list = (['air_conditioner', 'car_horn', 'children_playing',
                       'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
                       'jackhammer', 'siren', 'street_music'])
        label_list = sorted(label_list)
    else:

        # Build model and load in best weights
        model = build_crnn_onestep(n_freq_cnn=mel_bands)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        weightfile = os.path.join(expfolder, 'weights_best.hdf5'.format(expid))
        model.load_weights(weightfile)

        # Load data
        (x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test,
         id_test, label_list, scaler) = load_urbansed_crnn(
            sequence_frames=sequence_frames, sequence_hop=sequence_hop,
            sequence_offset=0,
            audio_hop=audio_hop, sr=sr,
            verbose=False, normalize=normalize_data,
            mel_bands=mel_bands)


        pred_test = model.predict(x_test)


    # ONLY KEEP SAMPLES THAT ARE INCLUDED IN THE FILE LIST!
    y_test_filtered = []
    pred_test_filtered = []
    for p, y, i in zip(pred_test, y_test, id_test):
        filename = os.path.basename(i[0]).replace('.npy.gz', '')
        if filename in file_list:
            y_test_filtered.append(y)
            pred_test_filtered.append(p)
    y_test_filtered = np.asarray(y_test_filtered)
    pred_test_filtered = np.asarray(pred_test_filtered)

    est_roll, ref_roll = combine_event_rolls(pred_test_filtered,
                                             y_test_filtered,
                                             threshold=0.5)

    # COMPUTE FOLD RESULTS AND REPORT
    # Convert event rolls into even lists
    est_event_list = event_roll_to_event_list(
        est_roll, label_list, audio_hop/float(sr))
    ref_event_list = event_roll_to_event_list(
        ref_roll, label_list, audio_hop/float(sr))

    # Compute metrics at 1s and 100ms levels
    seg_metrics1s = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=1.0)
    seg_metrics1s.evaluate(ref_event_list, est_event_list)
    results1s = seg_metrics1s.results()

    seg_metrics100ms = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=0.1)
    seg_metrics100ms.evaluate(ref_event_list, est_event_list)
    results100ms = seg_metrics100ms.results()

    fold_results.append([results1s, results100ms])

    # Report scores
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
    print(scores)

    # Save scores
    results_all = {'results1s': results1s, 'results100ms': results100ms}
    results_all_file = os.path.join(expfolder, 'test_results_file_list.json')
    json.dump(results_all, open(results_all_file, 'w'), indent=2)

    return results_all


def evaluate_per_file(expid, file_list, audio_hop=882, sr=44100,
                      sequence_frames=128,
                      sequence_hop=64, normalize_data=True, mel_bands=40,
                      load_predictions=True):

    modelfolder = (
        '/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/models')
    expfolder = os.path.join(modelfolder, expid)

    if load_predictions:

        ytestfile = os.path.join(expfolder, 'ytest.npy.gz')
        yprobfile = os.path.join(expfolder, 'yprob.npy.gz')
        yidfile = os.path.join(expfolder, 'yid.npy.gz')
        y_test = np.load(gzip.open(ytestfile, 'rb'))
        pred_test = np.load(gzip.open(yprobfile, 'rb'))
        id_test = np.load(gzip.open(yidfile, 'rb'))

        label_list = (['air_conditioner', 'car_horn', 'children_playing',
                       'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
                       'jackhammer', 'siren', 'street_music'])
        label_list = sorted(label_list)
    else:

        # Build model and load in best weights
        model = build_crnn_onestep(n_freq_cnn=mel_bands)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        weightfile = os.path.join(expfolder, 'weights_best.hdf5'.format(expid))
        model.load_weights(weightfile)

        # Load data
        (x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test,
         id_test, label_list, scaler) = load_urbansed_crnn(
            sequence_frames=sequence_frames, sequence_hop=sequence_hop,
            sequence_offset=0,
            audio_hop=audio_hop, sr=sr,
            verbose=False, normalize=normalize_data,
            mel_bands=mel_bands)


        pred_test = model.predict(x_test)


    # ONLY KEEP SAMPLES THAT ARE INCLUDED IN THE FILE LIST!
    y_test_filtered = []
    pred_test_filtered = []
    id_test_filtered = []
    for p, y, i in zip(pred_test, y_test, id_test):
        filename = os.path.basename(i[0]).replace('.npy.gz', '')
        if filename in file_list:
            y_test_filtered.append(y)
            pred_test_filtered.append(p)
            id_test_filtered.append(i)
    y_test_filtered = np.asarray(y_test_filtered)
    pred_test_filtered = np.asarray(pred_test_filtered)
    id_test_filtered = np.asarray(id_test_filtered)

    SegMetrics1s = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=1.0)
    SegMetrics100ms = sed_eval.sound_event.SegmentBasedMetrics(
        label_list, time_resolution=0.1)

    # DO EVAL FILE BY FILES
    idx = 0
    while idx < len(id_test_filtered):

        y_test_file = []
        pred_test_file = []
        id_test_file = []

        current_id = id_test_filtered[idx][0]
        while idx < len(id_test_filtered) and id_test_filtered[idx][
            0] == current_id:
            y_test_file.append(y_test_filtered[idx])
            pred_test_file.append(pred_test_filtered[idx])
            id_test_file.append(id_test_filtered[idx])
            idx += 1
        y_test_file = np.asarray(y_test_file)
        pred_test_file = np.asarray(pred_test_file)
        id_test_file = np.asarray(id_test_file)

        est_roll, ref_roll = combine_event_rolls(pred_test_file,
                                                 y_test_file,
                                                 threshold=0.5)

        # COMPUTE FOLD RESULTS AND REPORT
        # Convert event rolls into even lists
        est_event_list = event_roll_to_event_list(
            est_roll, label_list, audio_hop/float(sr))
        ref_event_list = event_roll_to_event_list(
            ref_roll, label_list, audio_hop/float(sr))

        # Compute metrics at 1s and 100ms levels
        SegMetrics1s.evaluate(ref_event_list, est_event_list)
        SegMetrics100ms.evaluate(ref_event_list, est_event_list)

    results1s = SegMetrics1s.results()
    results100ms = SegMetrics100ms.results()

    # Report scores
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
    print(scores)

    # Save scores
    results_all = {'results1s': results1s, 'results100ms': results100ms}
    results_all_file = os.path.join(expfolder, 'test_results_per_file.json')
    json.dump(results_all, open(results_all_file, 'w'), indent=2)

    return results_all
