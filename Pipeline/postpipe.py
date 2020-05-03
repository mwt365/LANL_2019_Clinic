#!/usr/bin/env python3
# coding:utf-8

"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Process a set of .dig files
  Created: 04/27/20

  Routines to process results after all segments
"""


import pandas as pd
import os
from pnspipe import process_command_file


def post_pipe(results, args):
    """
    Given a dictionary of results dictionaries and a list of
    commands (with arguments), process the commands.
    """

    # First organize the results by base file
    # The keys are the df.title fields, which may have one or more
    # slashes. If we are doing segments, we want the penultimate 2.
    # That's all we really should be doing, so let's go with that.
    # Prepare the sources dictionary whose keys are the files
    # and whose values are dictionaries with key
    # sources = [file][segment][results]

    sources = dict()
    for key in results.keys():
        fields = key.split('/')
        source, segment = fields[-2:]
        if source not in sources:
            sources[source] = dict()
        sources[source][segment] = results[key]

    # Now we can churn through the source files
    # Let's first decode the commands

    command_file = args.post
    if not os.path.exists(command_file):
        print(f"I could not find a file named {command_file} for post processing")
    else:
        orders = process_command_file(command_file)
        for fname, source in sources.items():
            segments = sorted(list(source.keys()))
            for order in orders:
                routine, kwargs = order
                routine(fname, source, segments, **kwargs)


def noise_stats(filename: str, source: dict, segments: dict, **kwargs):
    """
    Write this
    """
    fields = "figname;beta;lam1;lam2;amp;mean;stdev;chisq;prob".split(';')
    data = {key: [] for key in fields}
    for n, seg in enumerate(segments):
        noises = source[seg]['noise']
        for k in data.keys():
            try:
                data[k].append(noises[k])
            except:
                if k == 'beta':
                    data[k].append(None)
                    data['lam1'].append(noises['lamb'])
                    data['lam2'].append(None)
                    data['stdev'].append(noises['mean'])
    figures = data.pop('figname')
    df = pd.DataFrame(data)
    print(df.to_string(sparsify=False))
    df.to_csv(f"{filename}-noise.csv")

    # Now produce a page showing all the figures
    from Pipeline.figure_page import figure_page
    figure_page(f"{filename}-figs", figures)


def stats(filename: str, source: dict, segments: dict, **kwargs):
    """
    What does this really do?
    """
    fields = "figname;beta;lam1;lam2;amp;mean;stdev".split(';')
    data = {key: [] for key in fields}
    for n, seg in enumerate(segments):
        noises = seg['noise']
        for k in data.keys():
            data[k].append(noises[k])
    df = pd.DataFrame(data)
    print(df.to_string(sparsify=False))
    df.to_csv(f"{filename}-noise.csv")


