#jsonDriver.py
import json


# file = '/home/lanl/Documents/max/LANL_2019_Clinic/CH_1.json'

class JsonReadDriver:
    def __init__(self,file_name):

        with open(file_name, "r") as read_file:
            self.data = json.load(read_file)

    def getManualStart(self,file):
        return self.data['manual_starts'][file]

class JsonWriteDriver:
    def __init__(self,file_name):
        self.file_name = file_name
        self.datastore = {}

    def store_time_length(self,spec_file,length):
        self.datastore[spec_file] = length
    
    def flush(self):
        with open(self.file_name, 'w') as f:
            json.dump(self.datastore, f,indent=1)
