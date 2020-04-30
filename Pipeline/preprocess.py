#!/usr/bin/env python3
# coding:utf-8

"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Process a set of .dig files
  Created: 04/28/20

  Routines to act prior to running pipelines
"""

from ProcessingAlgorithms.preprocess.digfile import DigFile
from ProcessingAlgorithms.preprocess.fiducials import Fiducials
from ProcessingAlgorithms.preprocess.evenSplit import splitIntoEvenFrames
from re import Pattern
from os.path import join


def segment(inc_regex: Pattern, ex_regex: Pattern, **kwargs):
    """
    Process any files that need to be separated into
    segments, consistent with the include and exclude
    regular expressions. If the keyword argument
    'force' is True, then files that already have
    been segmented are resegmented.
    """
    force = kwargs.get('force', False)
    for dfname in DigFile.all_dig_files():
        # is this name consistent with the patterns?
        if not inc_regex.search(dfname):
            continue
        if ex_regex and ex_regex.search(dfname):
            continue
        df = DigFile(join(DigFile.dig_dir(), dfname))
        if not df.is_segment:
            n = df.has_segments
            if n > 0 and not force:
                continue
            # Now attempt to segment
            fids = Fiducials(df)
            splits = fids.values
            if len(splits):
                fids.split()
                print(f"Split {df.filename} into {len(splits)} segments using Fiducials")
                continue
            # If that didn't work, what else do we want to try?
            dt = kwargs.get('frame_length', 50e-6)  # 50 µs
            splitIntoEvenFrames(df, timeBetweenFrames=dt)
            print(f"Split {df.filename} into even frames")
