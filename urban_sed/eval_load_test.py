# CREATED: 4/24/17 15:35 by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
from scaper_waspaa2017.urban_sed.model import build_custom_cnn
from scaper_waspaa2017.urban_sed.data import load_urbansed_cnn_test
from scaper_waspaa2017.urban_sed.util import event_roll_to_event_list
from scaper_waspaa2017.urban_sed.util import combine_event_rolls
import sed_eval
import os
import json
import gzip

def evaluate_load_test(expid, audio_hop=882, sr=44100, sequence_frames=50,
                       sequence_hop=25, normalize_data=True, mel_bands=128,
                       large_cnn=False, frame_level_y=False):

    fold_results = []

    modelfolder = (
        '/scratch/js7561/datasets/scaper_waspaa2017/urban-sed/models')
    expfolder = os.path.join(modelfolder, expid)

    # Build model and load in best weights
    model = build_custom_cnn(n_freq_cnn=mel_bands, large_cnn=large_cnn)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    weightfile = os.path.join(expfolder, 'weights_best.hdf5'.format(expid))
    model.load_weights(weightfile)

    # Load data
    (x_test, y_test, id_test, label_list, scaler) = load_urbansed_cnn_test(
        sequence_frames=sequence_frames, sequence_hop=sequence_hop,
        audio_hop=audio_hop, sr=sr,
        verbose=False, normalize=normalize_data,
        mel_bands=mel_bands,
        frame_level_y=frame_level_y)

    pred_test = model.predict(x_test)

    if frame_level_y:
        est_roll, ref_roll = combine_event_rolls(pred_test, y_test,
                                                 threshold=0.5)
    else:
        ref_roll = y_test[:]
        est_roll = 1 * (pred_test >= 0.5)

    # COMPUTE FOLD RESULTS AND REPORT
    # Convert event rolls into even lists
    if frame_level_y:
        est_event_list = event_roll_to_event_list(
            est_roll, label_list, audio_hop/float(sr))
        ref_event_list = event_roll_to_event_list(
            ref_roll, label_list, audio_hop/float(sr))
    else:
        est_event_list = event_roll_to_event_list(
            est_roll, label_list, 1.0)
        ref_event_list = event_roll_to_event_list(
            ref_roll, label_list, 1.0)

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
    results_all_file = os.path.join(expfolder, 'test_results.json')
    json.dump(results_all, open(results_all_file, 'w'), indent=2)

    return results_all
