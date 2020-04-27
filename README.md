# LANL_2019_Clinic

This project is the product of the Los Alamos National Lab 2020 Harvey Mudd Clinic Team

## Purpose

The purpose of this project is to analyze PDV data, and generate useful information about the traces contained in them.

## Getting Started

### Python 3.6+ Required

### Required packages

Use the package manager (either pip or conda is compatible) to install required packages. From the root directory of the project run:


```bash
pip install -r requirements.txt
or
conda install --file requirements.txt
```
 
 ## Usage
 The project can be used in a variety off ways, however the intention is to use the `Spectrogram` class.

 ### `Spectrogram` class
 From a file or the python interpreter, this is invoked by creating an instance of the class.

 ```python
 spec = Spectrogram("PDV.dig")
 ```

 Many more options and optional arguments are available, as described in the documentation.


### Pipeline file
From `pipeline.py` batches of files can be processed at the same time, and the traces extracted. To do this:

1) Create a subdirectory with only the `.dig` files to be extracted.
2) Create a JSON file with the names and start points of the files.
3) Execute 
```python
python pipeline.py --json_name=NAME_OF_JSON_FILE_
```

The output of this program will be stored in a direcotry. More information can be found in the documentation

##Authors

* **Nick Koskelo** - *Project Manager*
* **Max Treutelaar**
* **Trevor Walker**
* **Rikki Walters** - *Spring Member*
* **Isabel Duan** - *Spring Member*
* **Professor Peter Saeta** -*Faculty Advisor*