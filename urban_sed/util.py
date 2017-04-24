# CREATED: 4/7/17 19:05 by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np


def contiguous_regions(act):
    act = np.asarray(act)
    onsets = np.where(np.diff(act) == 1)[0] + 1
    offsets = np.where(np.diff(act) == -1)[0] + 1

    # SPECIAL CASES
    # If there are no onsets and no offsets (all of act is the same value)
    if len(onsets) == 0 and len(offsets) == 0:
        if act[0] == 0:
            return np.asarray([])
        else:
            return np.asarray([[0, len(act)]])

    # If there are no onsets
    if len(onsets) == 0 and len(offsets) != 0:
        onsets = np.insert(onsets, 0, 0)

    # If there are no offsets
    if len(onsets) != 0 and len(offsets) == 0:
        offsets = np.insert(offsets, len(offsets), len(act))

    # If there's an onset before an offset, first onset is frame 0
    if onsets[0] > offsets[0]:
        onsets = np.insert(onsets, 0, 0)

    # If there's an onset after the last offset, then we need to add an offset
    # Offset is last index of activation (so that gives inverse of sed_eval)
    if onsets[-1] > offsets[-1]:
        offsets = np.insert(offsets, len(offsets), len(act))

    assert len(onsets) == len(offsets)
    assert (onsets <= offsets).all()
    return np.asarray([onsets, offsets]).T


def event_roll_to_event_list(event_roll, event_label_list, time_resolution):
    event_list = []
    for event_id, event_label in enumerate(event_label_list):
        event_activity = event_roll[:, event_id]
        event_segments = contiguous_regions(event_activity) * time_resolution
        for event in event_segments:
            event_list.append(
                    {'event_onset': event[0],
                     'event_offset': event[1],
                     'event_label': event_label})

    return event_list


def combine_event_rolls_bytrack(est_prob, ref_roll, id_matrix,
                                threshold=0.5):
    p_track_all = []
    y_track_all = []
    id_track_all = []
    id_prev = None

    for p, y, fid in zip(est_prob, ref_roll, id_matrix):

        pbin = (1 * (p >= threshold))
        if fid[0] == id_prev:
            p_track.extend(pbin.tolist())
            y_track.extend(y.tolist())
        else:
            if id_prev is not None:
                p_track_all.append(np.asarray(p_track))
                y_track_all.append(np.asarray(y_track))
                id_track_all.append(id_track)

            p_track = pbin.tolist()
            y_track = y.tolist()
            id_track = fid

            id_prev = fid[0]

    p_track_all = np.asarray(p_track_all)
    y_track_all = np.asarray(y_track_all)
    id_track_all = np.asarray(id_track_all)

    return p_track_all, y_track_all, id_track_all


def combine_event_rolls(est_prob, ref_roll, threshold=0.5):

    p_track = []
    y_track = []

    for p, y in zip(est_prob, ref_roll):
        pbin = (1 * (p >= threshold))
        p_track.extend(pbin.tolist())
        y_track.extend(y.tolist())

    p_track = np.asarray(p_track)
    y_track = np.asarray(y_track)

    return p_track, y_track
