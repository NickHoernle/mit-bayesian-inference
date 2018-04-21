'''
classification_evaluation.py
Author: Nicholas Hoernle
Date: 20 April 2018
Given a switching state space model (i.e. with known regime labels) and an inference scheme, I'd like to evaluate how well the inferred labels capture the true labels that are present in the data.
'''

import numpy as np
import copy

def hamming(seq1, seq2):
    """Calculate the Hamming distance between two arrays of labels"""
    assert len(seq1) == len(seq2)
    return np.sum(np.array(seq1, dtype=np.int16) != np.array(seq2, dtype=np.int16))

def match_labels_greedily(seq1, seq2):
    '''
    Function to return a mapping of the arguments of label string2 so that they most likely match those of label string1
    '''
    seq1 = np.array(seq1, dtype=np.int16)
    seq2 = np.array(seq2, dtype=np.int16)

    counts = np.bincount(seq1)
    most_freq = np.argsort(counts)[::-1]
    # what is the argument mapping that we use in this transformation (to recover the original learnt params)
    arg_map = {}

    for to_match in most_freq:
        indexes = np.where(seq1 == to_match)[0]

        if len(indexes) > 0:
            potential_matches = seq2[indexes]
            counts = np.bincount(potential_matches)

            matched = np.argmax(counts)

            if matched not in arg_map:
                arg_map[matched] = to_match

    to_merge = {}
    for k,v in arg_map.items():
        if v not in arg_map:
            to_merge[v] = k

    arg_map = {**arg_map, **to_merge}

    for i in range(0, np.max(seq2) + 1):
        if i not in arg_map:
            arg_map[i] = i
    return arg_map

def update_label(seq1, seq2, theta_params=[]):
    '''
    update the labels of seq2 to match the labels of seq1
    '''
    assert(len(seq1) == len(seq2))

    update_dict = match_labels_greedily(seq1, seq2)
    new_arr = np.array([update_dict[x] for x in seq2])

    new_thetas = np.array([[] for i in range(len(theta_params))])
    if len(theta_params) > 0:

        for i, theta in theta_params:
            new_thetas[update_dict[i]] = theta

    return new_arr, new_thetas

def get_hamming_distance(seq1, seq2):
    '''
    calculate the hamming distance between two sequences of labels.
    '''
    seq2_updated, sorted_thetas =  update_label(copy.deepcopy(seq1), copy.deepcopy(seq2))
    return seq2_updated, sorted_thetas, hamming(seq1, seq2_updated)
