import random
import spectrogram as spctgrm
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy import signal


"""

1. Generate fake arrays for testing.
2. Find the maximum n intensities in the first arrays, return their indicies and intensity values.
3. Find minimum energy paths for all high intensity signals up to stop time. 

4. Store those indicies and intensities in DP table?

5. For each potential seam in the DP table, find the change in intensity for each time step of experiment.
6. If the intensity deviates too far from the starting intensity, then that seam can not be a baseline. 
7. Return intensity value for all potential baselines to the user for selection. 
8. Do fancy things with the intensity to find errors. 

"""


def generate_random_array(test_range):

    outer_array = []

    for i in range(test_range):

        inner_array = [round(random.random(), 5) for i in range(test_range)]

        outer_array.append(inner_array)

    return outer_array



def find_potential_seams(two_dimensional_array, num_signals: int):

    # assert isinstance(two_dimensional_array, list)
    # assert isinstance(two_dimensional_array[0], list)
    assert isinstance(two_dimensional_array[0][0], float)

    # for row in two_dimensional_array : 
        # print(row) 
    # rez = [[two_dimensional_array[j][i] for j in range(len(two_dimensional_array))] for i in range(len(two_dimensional_array[0]))] 
    # print("\n") 
    # for row in rez: 
        # print(row) 

    outer_array = len(two_dimensional_array)
    first_array = two_dimensional_array[20]
    inner_array = len(first_array)

    new_first_array = [(first_array[i], i) for i in range(len(first_array))]
    new_first_array.sort(key=lambda tup: tup[0], reverse=True)
    # print(new_first_array)

    potential_baseline_tuples = [new_first_array[i] for i in range(num_signals)]
    potential_baseline_intensities = [i[0] for i in potential_baseline_tuples]
    potential_baseline_indicies = [i[1] for i in potential_baseline_tuples]
    # print()
    # for row in two_dimensional_array:
    #     print(row)
    # print()
    # print(potential_baseline_intensities)
    # print(potential_baseline_indicies)

    return potential_baseline_indicies


def find_seams(two_dimensional_array: list, num_signals: int, t_end:int):

    # assert isinstance(two_dimensional_array, list)
    # assert isinstance(two_dimensional_array[0], list)
    assert isinstance(two_dimensional_array[0][0], float)

    rez = [[two_dimensional_array[j][i] for j in range(len(two_dimensional_array))] for i in range(len(two_dimensional_array[0]))] 

    potential_seams = find_potential_seams(rez, num_signals)

    # DP = [[] for i in range(len(rez))]

    seams = []
    new_seams = []

    for start_of_seam in potential_seams:

        # print(start_of_seam)
        # print(rez[0][start_of_seam], "\n")

        # if find_seam(start_of_seam, two_dimensional_array, DP):

        seam_trace = []


        if find_seam(start_of_seam, 0, rez, seam_trace, t_end):

            seams.append(seam_trace)
    
    # for seam in seams:

    #     if verify_seam(seam):

    #         new_seams.append(seam)

    # print(len(seams))
    # print(len(new_seams))

    return seams



"""
time: 0 ----> end of experiment
velocity: min ---> max
seam_trace: array that starts empty and keeps track of coordinates for max values

two-dim-array: intensity @ velo and time = two_dimensional_array[time][velocity]

"""

def find_seam(velocity:int, time:int, two_dimensional_array:list, seam_trace:list, limit:int):

    if time > limit:
        return seam_trace

    current_intensity = two_dimensional_array[time][velocity]

    if time+1 > limit:
        seam_trace.append((velocity,time,current_intensity))
        return seam_trace


    # for row in two_dimensional_array:
    #     print(row)
    # print("velocity at time %d: %d" %(time, velocity))
    # print("intensity: ", current_intensity)

    velo_above = velocity+1
    velo_below = velocity-1

    change_velo_above = float("inf")
    change_same_velo = float("inf")
    change_velo_below = float("inf")

    if velo_above < len(two_dimensional_array[0]):
        change_velo_above = abs(current_intensity - two_dimensional_array[time+1][velocity+1])
    if velo_below > -1:
        change_velo_below = abs(current_intensity - two_dimensional_array[time+1][velocity-1])

    change_same_velo = current_intensity - two_dimensional_array[time+1][velocity]
    
    # print(abs(change_velo_below), abs(change_same_velo), abs(change_velo_above))
    # print()

    minimum_change = min(abs(change_velo_below), change_same_velo, change_velo_above)
    seam_trace.append((velocity,time,current_intensity))

    if minimum_change == change_velo_above:
        return find_seam(velocity+1, time+1, two_dimensional_array, seam_trace, limit)
    elif minimum_change == change_same_velo:
        return find_seam(velocity, time+1, two_dimensional_array, seam_trace, limit)
    else:
        return find_seam(velocity-1, time+1, two_dimensional_array, seam_trace, limit)






def verify_seam(seam):

    seam_length = len(seam)

    seam_start = round(seam_length * .01)

    start_sum = 0

    for i in range(seam_start):

        start_sum += seam[i][2]
    
    # average_start = start_sum / seam_start
    average_start = seam[1][2]

    for tup in seam:

        if within_range(tup[2], 50, average_start) is False:

            return False

    return True




def within_range(test, percent, base):
    decimalPercent = percent / 200.0
    highRange = base * (1.0 + decimalPercent)
    lowRange = base * (1.0 - decimalPercent)

    return lowRange <= test and test <= highRange





if __name__ == "__main__":

    sp_nl = spctgrm.Spectrogram('GEN3CH_4_009.dig')
    sgram_nl = sp_nl.spectrogram_no_log(0, 50e-6)

    # intensities = [np.sum(sgram_nl['spectrogram'], axis=1)]
    intensities = []

    for inner_array in sgram_nl['spectrogram']:
        max_intensity = np.max(inner_array)
        intensities.append(max_intensity)

    max_intensity = np.max(intensities)

    # baseline_velocity_index = [np.argmax(intensities)]
    # baseline_velocity = sgram_nl['v'][baseline_velocity_index]

    print(max_intensity)
    print("\n")
    print("\n")

    # print(sgram_nl['spectrogram'].shape)

    # print(sgram_nl['v'])
    # print(sgram_nl['t'])
    # print(sgram_nl['spectrogram'])

    sp = sgram_nl['spectrogram']

    # indices = find_potential_seams(sp, 30)
    # print(indices)

    time_length = len(sgram_nl['t'])

    # mock_intensities = generate_random_array(array_dimensions)

    potential_baselines = find_seams(sp, 20, time_length-200)

    # print(potential_baselines[0][0])

    baseline_velocity_index = potential_baselines[0][0][0]
    baseline_intensity = potential_baselines[0][0][2]

    baseline_velocity = sgram_nl['v'][baseline_velocity_index]

    print("baseline velocity: ", baseline_velocity)
    print("baseline intensity: ", baseline_intensity)

    # print(sgram_nl['v'])
    

        
