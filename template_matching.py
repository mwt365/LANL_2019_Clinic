#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: To automatically find desired regions in 
            an intensity matrix that match specified templates.
  Created: 1/28/2020
"""

import numpy as np
import pandas as pd
from spectrogram import Spectrogram
import peak_follower
import baselines as bls


class Template:
    """
    Constructor for Template Objects.

    Inputs:
      -  width: the width of a template
      -  height: the height of a template
      -  values: the 2-dimensional array of weights for the template

    Outputs:
      -  template object: This will be used to calculate a score for the
                            region it is placed on in the spectrogram.

    Observations:
        Using multiple templates for a combined score has proven to work 
        marginally better than a singular template. The template values
        are multiplied by their coresponding intensity values in the spectrogram,
        and then summed to get a total score for that region. Some templates have
        been pre-made below.
        
    """

    def __init__(self,
                 width=None,
                 height=None,
                 values=None
                 ):

        if values != None:
            self.values = values
            self.width = len(values[0])
            self.height = len(values)
        else:
            self.width = width if width != None else 0
            self.height = height if height != None else 0
            self.values = None


start_pattern = [
            [-1, -1, -1, -1],
            [-1, -1, 5,  5 ],
            [-1, -1, -1, -1]]

start_pattern2 = [
            [-1, -1, -1, 4],
            [-1, -1, 4,  4],
            [-1, -1, -1,-1]]

start_pattern3 = [
            [-1, -1, -1, 3],
            [-1, -1, 3,  3],
            [-1, -1, -1, -1]]

start_pattern4 = [
            [-3, -1, 1, 2],
            [-3, -1, 1, 2],
            [-3, -1, 1, 2]]

start_pattern5 = [
            [-1, 1, 1, 2],
            [-1, -1, 1, 1],
            [-3, -1, -1, 1]]





def calculate_score(velo_index, template, intensities, time_index):
    """
    Returns the score for the template on a certain region in the
    spectrogram. The score is the sum of all products of the template 
    values and their corresponding intensity values. 

    Inputs:
      -  velo_index: the velocity index of the spectrogram that the template
                    will start at when computing the sum of all products. 
      -  template: a template object with values, a width, and a height. 
      -  intensities: a 2 dimensional array of values that are produced by a 
                    rolling fast fourier transform of voltage data. See
                    Spectrogram.py for more details. 
      - time_index: the time offset specified by the user to index into the 
                    2 dimensional array of intensities. 

    Outputs:
      -  template_sum: the sum of all inner products between intensities and 
                    template values. 

    """

    template_sum = 0        

    for values in template.values:

        for i, value in enumerate(values):

            template_sum += value * intensities[velo_index][time_index+i]

    return template_sum



def find_potential_baselines(sgram):
    """
    Returns a list of velocity values that correspond to the start of 
    a potential baseline in the spectrogram. The user will be asked to 
    verify that the baseline found is accurate, and indeed a baseline. 

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.

    Outputs:
      -  new_baselines: a list of velocity values that correspond to the start of 
                a potential baseline in the input spectrogram. 

    """
    #TODO replace the current baseline with baselines_by_squash
    #TODO use it with peak follower
    #TODO expand templates and normalization
    #TODO dot product between one dimensional vectors using .flatten
    #TODO expand to cover crossings and other phenomena

    baselines = []
    hoods = bls.baselines_by_fft(sgram)

    for n, h in enumerate(hoods):
        max_v = 0
        max_i = 0
        # print(f"Peak {n}\nVelocity{n}\tIntensity{n}")
        v, i = h
        for j in range(len(v)):
            # print(f"{v[j]:.4f}\t{i[j]:.4f}")
            if i[j] > max_i:
                max_i = i[j]
                max_v = v[j]    
        baselines.append(max_v)

    new_baselines = []

    # new_baselines.append(baselines[1])

    for baseline in baselines:
        print("Is there a baseline at: ", baseline, "?", end=" ")
        ans = input("(y/n)\n")
        if ans == "y":
            new_baselines.append(baseline)
        else:
            continue

    return new_baselines



def find_start_time(sgram):
    """
    Returns a potential jump off time for each baseline after 
    prompting the user for that input. 

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.

    Outputs:
      -  time_index: the start time index that can be used to find actual 
                times in the spectrogram.time array. 

    """

    time_index = None
    # start_time = 12 * 10**-6
    # time_index = sgram._time_to_index(start_time)

    ans = input("Where does the start begin? (in microseconds)\n")
    try:
        start_time = int(ans) * 10**-6
        # baseline_index = sgram._velocity_to_index(baseline)
        time_index = sgram._time_to_index(start_time)
    except:
        print("Input can not be converted to integer")
        return None

    return time_index



def find_regions(sgram, templates, velo_bounds=None, time_bounds=None):
    """
    Returns a dictionary of dictionaries. The outermost dictionary keys are 
    time values. Those time values each have they're own dictionaries, with 
    velocity keys corresponding to the score calculated with that key's velocity
    as the starting position for each of the templates.

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.
      - templates: a list of template objects.
      - velo bounds: will keep the algorithm from searching outside of certain velocity ranges
      - time bounds: will keep the algorithm from searching outside of certain time ranges

    Outputs:
      -  all_scoures: a dictionary of dictionaries. Which contain the scores from the inputed 
                    templates and spectrogram.intensity matrix.  
    """

    if velo_bounds is not None:
        assert isinstance(velo_bounds, tuple)
        lower_velo = velo_bounds[0]
        upper_velo = velo_bounds[1]
    elif velo_bounds is None:
        upper_velo_index = sgram.intensity.shape[0]
        upper_velo = sgram.velocity[upper_velo_index]
        lower_velo = sgram.velocity[0]
    if time_bounds is not None:
        assert isinstance(time_bounds, tuple)
        lower_time = time_bounds[0]
        upper_time = time_bounds[1]
    elif time_bounds is None:
        upper_time_index = sgram.intensity.shape[1]
        upper_time = sgram.time[upper_time_index]
        lower_time = sgram.time[0]

    baselines = find_potential_baselines(sgram)
    time_index = find_start_time(sgram)

    if time_index is None:
        return

    upper_velo_index = sgram._velocity_to_index(upper_velo)
    lower_velo_index = sgram._velocity_to_index(lower_velo)

    time_max = sgram.intensity.shape[1]
    velocity_max = sgram.intensity.shape[0]

    # print(time_max)
    # print(velocity_max)
    # print("upper: ",upper_velo_index)
    # print("lower: ",lower_velo_index)
    # print(sgram.intensity)
    # print(template.values)
    # print(time_index)

    all_scores = {}


    max_time = sgram.intensity.shape[1]

    if len(templates) > 1:
        width = templates[0].width
    else:
        width = templates.width

    start_index = time_index - (2*width)
    end_index = time_index + (2*width)

    if start_index < 0:
        start_index = 0
        end_index = width*4
    if end_index+width > max_time:
        end_index = max_time-width-1

    for i in range(start_index, end_index, 1):

        scores = {}

        for velocity_index in range(upper_velo_index, lower_velo_index, -1):

            score = 0

            for template in templates:

                score += calculate_score(velocity_index, template, sgram.intensity, i)
            
            scores[velocity_index] = score
        
        all_scores[i] = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
        
    return all_scores



def find_potenital_start_points(sgram, all_scores):
    """
    Returns a list of interesting coordinates in the spectrogram
    that can correspond to potential jumpoff points. These 
    points are likely near the actual starting 
    jumpoff point. 

    Inputs:
      -  sgram: the spectrogram object. It has time values, velocities, and 
                a 2 dimensional array of intensities. See spectrogram.py for
                more details.
      - all_scores: a dictionary of dictionaries.

    Outputs:
      -  interesting_points: a list of (time, velocity) tuples.  
    """

    temp = []

    for time in all_scores.keys():

        keys = list(all_scores[time].keys())
        indicies = []

        for i in range(len(keys)-1, len(keys)-5, -1):
            tup = (time, keys[i])
            indicies.append(tup)

        # print(time,'\n')
        # print(all_scores[time][i])
        # print(indicies,'\n')

        for i in indicies:
            tup = (i, sgram.velocity[i[1]], all_scores[i[0]][i[1]])
            temp.append(tup)
    
    final = sorted(temp, key = lambda x: x[2], reverse=True)

    interesting_points = []

    for i in range(10):

        tup, velo, score = final[i]
        t, v = tup

        # print("velocity: ", velo)
        # print("time: ", sgram.time[t])
        # print("score: ", score, '\n')
        
        interesting_points.append((sgram.time[t], velo))

    return interesting_points


def get_bounds_from_user():
    """
    Prompts that user for bounds on velocity and time to make
    searching easier for the algorithms (not to mention faster).

    Outputs:
      -  time_bounds: a tuple of time values.  
      -  velo_bounds: a tuple of velocity values. 
    """

    lower_bound_t = input("Enter a time to the left of the jumpoff point: \n")
    lower_bound_t = int(lower_bound_t) * 10**-6
    upper_bound_t = input("Enter a time to the right of the jumpoff point: \n")
    upper_bound_t = int(upper_bound_t) * 10**-6

    upper_bound_v = input("Enter a velocity above the jumpoff point: \n")
    upper_bound_v = int(upper_bound_v)
    lower_bound_v = input("Enter a velocity below the jumpoff point: \n")
    lower_bound_v = int(lower_bound_v)

    # lower_bound_t = 10 * 10**-6
    # upper_bound_t = 14 * 10**-6
    # upper_bound_v = 3700
    # lower_bound_v = 2000


    time_bounds = (lower_bound_t, upper_bound_t)
    velo_bounds = (lower_bound_v, upper_bound_v)

    return time_bounds, velo_bounds



if __name__ == '__main__':
    import os
    from digfile import DigFile

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs"
    os.chdir(path)
    df = DigFile('CH_2_009.dig')


    sgram = Spectrogram(df, 0.0, 60.0e-6, form='db')


    time_bounds, velo_bounds = get_bounds_from_user()


    template = Template(values=start_pattern)
    template2 = Template(values=start_pattern2)
    template3 = Template(values=start_pattern3)
    template4 = Template(values=start_pattern4)


    templates = [template, template2, template3, template4]


    scores = find_regions(sgram, templates, velo_bounds, time_bounds)

    interesting_points = find_potenital_start_points(sgram, scores)

    # print(interesting_points, '\n')

    total_time = 0
    total_velo = 0

    for i in interesting_points:

        time, velo = i

        total_time += time
        total_velo += velo

    average_time = total_time / len(interesting_points)
    average_velo = total_velo / len(interesting_points)

    print(average_time)
    # print(average_velo)

