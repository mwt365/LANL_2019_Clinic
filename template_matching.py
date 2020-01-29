
import numpy as np
from spectrogram import Spectrogram
import peak_follower
import baselines as bls


class Template:

    def __init__(self,
                 width=None,
                 height=None,
                 values=None
                 ):

        self.width = width if width != None else 0
        self.height = height if height != None else 0
        self.values = values if values != None else []


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
            [-1, -1, -1, 3]]

start_pattern4 = [
            [-3, -2, -1, 2],
            [-3, -1, 2,  3],
            [-3, -2, -1, 2]]



def calculate_score(index, template, intensities, time_index):

    template_sum = 0        

    for values in template.values:

        for i, value in enumerate(values):

            template_sum += value * intensities[index][time_index+i]

    return template_sum



def find_potential_baselines(sgram):

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



def find_start_time(sgram, baselines):

    time_index = None

    for baseline in baselines:

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
    time_index = find_start_time(sgram, baselines)

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

    for i in range(30):

        tup, velo, score = final[i]
        t, v = tup

        # print("velocity: ", velo)
        # print("time: ", sgram.time[t])
        # print("score: ", score, '\n')
        
        interesting_points.append((sgram.time[t], velo))

    return interesting_points



if __name__ == '__main__':
    import os
    from digfile import DigFile

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs"
    os.chdir(path)
    df = DigFile('CH_2_009.dig')

    baselines_v = []

    sgram = Spectrogram(df, 0.0, 60.0e-6, form='db')

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

    width = 4
    height = 3

    template = Template(width, height, start_pattern)
    template2 = Template(width, height, start_pattern2)
    template3 = Template(width, height, start_pattern3)
    template4 = Template(width, height, start_pattern4)


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

