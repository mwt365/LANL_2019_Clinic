# LANL_2019_Clinic


## Purpose
This repository is a tool used for analyzing voltage data from Photon Doppler Velocimetry experiments. Written in Python3, the routines included in this project will extract voltages from dig files, apply a fast Fourier transform to those values, and begin to extract velocity from surfaces in the experiment in reasonable time. Integrating this software into an autonomous system that could process many of these data files was goal of the developers constructing it.

# Overview
The general layout of the repository is separated into sections, depending on the type of desired analysis. There are a wide variety of methods in place to generate meaningful data about these files, ranging from the probalistic modeling of noise to using computer vision to find signal name a few. All are contained in their respective subfolders. 

**IMPORTANT**
*These files make an assumption as to what the file structure in the top level directory of the repository will look like, this will be explained further in the section below.*

This ***ReadMe*** is structured as follows:

- File Structure
- Data
- Code 
- Pipeline
- Dependencies


## File Structure

Since the folders holding the dig files and other important pieces of data exist at the same level as the cloned repository, there is a recommended way of structuring the files within the folder holding this repository.

In general, the folder holding the cloned repository was developed with this structure, illustrated below. It can be changed, however the paths used in many of the underlying must also reflect the flow of data and output to the log files in the script directory. 

```
PDV_project
│       
│
└───LANL_2019_Clinic
│   │   README.md // you are here!
│   │   spectrogram.py
│   │   baselines.py
│   │   pnspipe.py
│   │   spectrogram_widget.py
│   │   template_matcher.py
│   │   ...
│   │
│   └───ProcessingAlgorithms
│   │   │   spectrum.py
│   │   │   
│   │   └───preprocess
│   │       │   digfile.py
│   │       │   ...
│   │
│   └───ImageProcessing
│       │   ...
│       │
│       └───Templates
│           │  im_template.png
│           │  ...
│
└───jupyter // for holding jupyter notebooks
│   │   ...
│   │
│
└───script // this folder is used for the automated pipeline
│   │   script.txt
│   │   ...
│   
└───dig // this folder has all dig files
│   │   baselineTracking.csv
│   │   long_experiment.dig 
│   │
│   └───digfolder1
│       │   experiment1.dig
│       │   experiment2.dig
│       │   ... 
│
└───EstimateJumpOffPosition.xlsx
│
└───EstimateProbeDestruction.xlsx

```


## Data 

Since the data files are owned by the Los Alamos National Laboratory and is not cleared to be released to the public, we could not include them here in this repository. That being said,these files are often bulky, and can have billions of data points contained in them. So including them here on the online repository is not a viable option.

These files are generated from the oscilloscopes that are set up in the experiment. More specificallly, incoming light waves hit a sensor and are digitized over time where the data can be outputted in its raw form. These are the dig files that we receive and can process. 

## Code

#### Creating Spectrograms from dig files

The most important feature to the project is making a spectrogram that corresponds to the voltage data in a dig file. This can be done in the spectrogram.py file in the repository. To run this command from a terminal window, it would look something like this: 

```Python3
$ cd LANL_2019_clinic

$ python3 spectrogram.py
```

In the call to spectrogram, an object gets made that will hold the data computed from the fast Fourier transform. Where you can pass in a DigFile object from digfile.py or just specify the path to the dig file held in another directory. This is shown in both ways below. The transformed values would be saved in a data member called 'intensity' from the Spectrogram class. 

```
from ProcessingAlgorithms.preprocess.digfile import DigFile
df = DigFile('GEN3CH_4_009/seg00')
sp = Spectrogram(df, ending=35.0e-6)

### or 

df_path = "../dig/GEN3CH_4_009/seg00.dig"
sp = Spectrogram(df_path, 0.0, 60.0e-6, overlap_shift_factor= 7/8, form='db')
```

A lot of other analysis methods depend on a spectrogram object to be made, and often take in the made object as input. For example, template matching would take in a spectrogram object to make a template matching object, as shown below. 

```Python3
$ cd LANL_2019_clinic

$ python3 template_matcher.py
```

```
from ProcessingAlgorithms.preprocess.digfile import DigFile
from spectrogram import Spectrogram

df = DigFile('GEN3CH_4_009/seg00')

spec = Spectrogram(df, ending=35.0e-6)

tm = TemplateMatcher(spec, Templates.opencv_long_start_pattern5.value)

tm.match()
```

## Pipeline

One of the goals in this research project was to automate velocity extraction on a large set of data files. The idea of a pipeline is to send data through multiple different functions, logging results along the way to another location, in order to get an autonomously working system to output meaningful statistics about those files for a human to quickly read and understand what is going on. 

This is a feasible way to apply any number of functions to a large batch of files without any human interaction. We could do this by utilizing the code written in *pnspipe.py*. 

To run the pipeline, you must first move to the script folder in the top level directory. 

```
cd PDV_project/script/
open script.txt
```

What you should see is a plain text file that have some function names. These function names are methods specified in *pnspipe.py*, and are read and executed when the file is run. You could create custom functions that demonstrate the functionality of any of the repository methods. All you need to do is import the correct files and libraries to do so. An example of a custom function is shown below. In *pnspipe.py*:

```
def find_baselines(pipe: PNSPipe, **kwargs):
    """
    Compute baselines for pipe.spectrogram using the
    baselines_by_squash method. If baseline_limit is
    passed as a keyword argument, only keep baseline values
    that are larger than this value (which should be between
    0 and 1).
    """
    from baselines import baselines_by_squash as bline # import the function or library to use
    peaks, widths, heights = bline(pipe.spectrogram) # call the function on the saved pipe data
    baseline_limit = kwargs.get('baseline_limit', 0.01)
    pipe.baselines = peaks[heights > baseline_limit]
    blines = ", ".join([f"{x:.1f}" for x in pipe.baselines])
    pipe.log(f"Baselines > {baseline_limit*100}%: {blines}", True) # log the results to a file

```

In the plain text file, you need to tell *pnspipe* to run this specific function on the data. So it should look like this in script.txt.

```
find_baselines

```

Where there might be other functions that you would like to call along with finding the baselines of a specific file. To start the pipeline, make sure you are in the script directory and run the *pnspipe* file in the repository. 

```
$ cd PDV_project/script/
$ python3 ../LANL_2019_clinic/pnspipe.py --delete --regex=*.dig

```

This should first delete all existing log files in the script directory, and then start finding the baselines to all files that match the optional regular expression "*.dig". The log files should then populate the script directory, and have the same structure as the folder that has all of the dig files to be analyzed. 


## Dependencies

All dependencies should be specified in the requirements.txt file in the repository. 


## Collaborators

* Trevor Walker
* Nick Koskelo 
* Max Treutelaar 
* Rikki Walters
* Isabel Duan
* Peter Saeta
* Los Alamos National Laboratory
