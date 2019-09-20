#! /usr/bin/env python3

"""
Program for processing digilink files.
"""
import os


instrument_spec_codes = {
    'BYT_N': 'binary_data_field_width',
    'BIT_N': 'bits_per_trace_point',
    'ENC': 'encoding',
    'BN_F': 'binary_number_format',
    'BYT_O': 'byte_order',
    'WFI': 'source_trace',
    'NR_P': 'number_pixel_bins',
    'PT_F': 'point_format',
    'XUN': 'horizontal_units',
    'XIN': 'sample_interval',
    'XZE': 'post_trigger_seconds',
    'PT_O': 'pulse_train_output',
    'YUN': 'vertical_units',
    'YMU': 'vertical_scale_factor',
    'YOF': 'vertical_offset',
    'YZE': 'vertical_offset_y_component',
    'NR_FR': 'NR_FR'
}


def wave_spec_to_dict(filename):
    """
    This function is able to turn the header of
    digilink files into a python dictionary
    of potentially useful information regarding the
    waveform captured by the oscilloscope.
    """
    strings = instrument_spec_codes.keys()
    header_limit = 11
    current_line = 0
    f = open(filename, 'rb')
    lines_byte = f.readlines(256)
    lines_string = [line.decode("utf-8") for line in lines_byte]

    for line in lines_string:
        if any(s in line for s in strings):
            str(line)
            data = line.split(':')
            waveform_data = data[2].split(';')
            waveform_specs_dict = {}

            for element in (waveform_data):
                index = element.find(' ')
                value = element[index+1:]
                spec_code = element[:index]
                wave_spec_name = instrument_spec_codes[spec_code]
                waveform_specs_dict[wave_spec_name] = value

            return waveform_specs_dict
        current_line += 1
    f.close()


def preprocess(filename):
    """
    Add more to this function. Read in the
    Binary data for processing.
    """
    if filename is None:
        print("could not find file, exiting...")
        return None
    else:
        wave_info = wave_spec_to_dict(filename)
        print(wave_info)


def main():
    for file_ in os.listdir("."):
        if file_.endswith(".dig"):
            dig_file = file_
            preprocess(dig_file)


if __name__ == '__main__':

    main()
