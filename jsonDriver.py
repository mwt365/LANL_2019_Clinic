#jsonDriver.py
import json


# file = '/home/lanl/Documents/max/LANL_2019_Clinic/CH_1.json'

class JsonDriver:
    def __init__(self,file_name):

        with open(file_name, "r") as read_file:
            self.data = json.load(read_file)

    def getManualStart(self,file):
        return self.data['manual_starts'][file]