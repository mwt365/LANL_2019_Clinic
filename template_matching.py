
import numpy as np
from spectrogram import Spectrogram
import peak_follower
import baselines


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
            [-1, -1, -1, 3],
            [-1, -1, 3,  3],
            [-1, -1, -1,-1]]

start_pattern3 = [
            [-2, -1, -1, 3],
            [-2, -1, 3,  3],
            [-2, -1, -1,-1]]


def calculate_score(index, template, intensities, time_index):

    template_sum = 0 

    for values in template.values:

        for i, value in enumerate(values):

            template_sum += value * intensities[index][time_index+i]

    return template_sum



# def setup(path):

#     os.chdir(path)
#     df = DigFile('CH_2_009.dig')
#     baselines_v = []
#     sgram = Spectrogram(df, 0.0, 60.0e-6, form='db')
#     hoods = baselines.baselines_by_fft(sgram)

#     # lower_bound_t = input("enter a time to the left of the jumpoff point.")
#     # lower_bound_t = int(lower_bound_t) * 10**-6
#     # upper_bound_t = input("enter a time to the right of the jumpoff point.")
#     # upper_bound_t = int(upper_bound_t) * 10**-6

#     # upper_bound_v = input("enter a velocity above the jumpoff point.")
#     # upper_bound_v = int(upper_bound_v)
#     # lower_bound_v = input("enter a velocity below the jumpoff point.")
#     # lower_bound_v = int(lower_bound_v)

#     lower_bound_t = 10 * 10**-6
#     upper_bound_t = 14 * 10**-6
#     upper_bound_v = sgram._velocity_to_index(3500)
#     lower_bound_v = sgram._velocity_to_index(2000)

#     tvals, vvals, ivals = sgram.slice((lower_bound_t, upper_bound_t), (lower_bound_v, upper_bound_v))

#     for n, h in enumerate(hoods):
#         max_v = 0
#         max_i = 0
#         # print(f"Peak {n}\nVelocity{n}\tIntensity{n}")
#         v, i = h
#         for j in range(len(v)):
#             # print(f"{v[j]:.4f}\t{i[j]:.4f}")
#             if i[j] > max_i:
#                 max_i = i[j]
#                 max_v = v[j]    
#         # print("\n")
#         baselines_v.append(max_v)


#     actual_baselines = []

#     for baseline in baselines_v:
#         print("is there a baseline at: ", baseline, "?", end=" ")
#         ans = input("(y/n)\n")
#         if ans == "y":
#             actual_baselines.append(baseline)
#         else:
#             continue



if __name__ == '__main__':
    import os
    from digfile import DigFile

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs"
    os.chdir(path)
    df = DigFile('CH_2_009.dig')

    baselines_v = []

    sgram = Spectrogram(df, 0.0, 60.0e-6, form='db')
    hoods = baselines.baselines_by_fft(sgram)

    # lower_bound_t = input("enter a time to the left of the jumpoff point.")
    # lower_bound_t = int(lower_bound_t) * 10**-6
    # upper_bound_t = input("enter a time to the right of the jumpoff point.")
    # upper_bound_t = int(upper_bound_t) * 10**-6

    # upper_bound_v = input("enter a velocity above the jumpoff point.")
    # upper_bound_v = int(upper_bound_v)
    # lower_bound_v = input("enter a velocity below the jumpoff point.")
    # lower_bound_v = int(lower_bound_v)

    lower_bound_t = 10 * 10**-6
    upper_bound_t = 14 * 10**-6
    upper_bound_v = sgram._velocity_to_index(3500)
    lower_bound_v = sgram._velocity_to_index(2000)

    tvals, vvals, ivals = sgram.slice((lower_bound_t, upper_bound_t), (lower_bound_v, upper_bound_v))


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
        # print("\n")
        baselines_v.append(max_v)


    actual_baselines = []

    # for baseline in baselines_v:
    #     print("is there a baseline at: ", baseline, "?", end=" ")
    #     ans = input("(y/n)\n")
    #     if ans == "y":
    #         actual_baselines.append(baseline)
    #     else:
    #         continue
    
    actual_baselines.append(baselines_v[1])

    for baseline in actual_baselines:
        ans = input("where does the start begin? (microseconds)")
        try: 
            start_time = int(ans) * 10**-6
            # baseline_index = sgram._velocity_to_index(baseline)
            time_index = sgram._time_to_index(start_time)

        except:
            print("input can not be converted to integer")
            break

    width = 4
    height = 3

    newTemplate = Template(width, height, start_pattern3)

    time_max = sgram.intensity.shape[1]
    velocity_max = sgram.intensity.shape[0]

    scores = {}

    for velocity_index in range(upper_bound_v, lower_bound_v, -1):
        
        score = calculate_score(velocity_index, newTemplate, sgram.intensity, time_index)
        # score = calculate_score(velocity_index, newTemplate, ivals, time_index)
        scores[velocity_index] = score

    
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    indicies = list(sorted_scores.keys())

    for i in range(len(indicies)-1, len(indicies)-10, -1):
        velocity = sgram.velocity[indicies[i]]
        print("velocity: ",velocity)
        print("score: ", sorted_scores[indicies[i]],"\n")


    # print(start_time)
