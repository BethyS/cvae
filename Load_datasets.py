# -*- coding: utf-8 -*-
"""
This fuction loads sensor location data and measuments from each sesnsors.
This is the python version of "load_dataset.R"
@author: Bethelhem S

(Version 1.0.2, built: Wed Jul  7 15:45:17 2021)  

## ## Copyright (C)2020 Bethelhem SEifu
"""

## Import important Lib. 
import numpy 
import pandas
import os

def load_dataset(sensors_ID=None):
    if(sensors_ID is None):
        Sensor_filename = "D:/New folder/Bety Python/Data/Image_data/Sensor_List_new.csv"
        sensors= pandas.read_csv(Sensor_filename, header=0, index_col=None,low_memory=False)
        sensors = sensors[sensors['country']=='Spain']
        sensors_ID= pandas.DataFrame(sensors.id)
        sensors_ID=sensors_ID.reset_index()
    series=  [pandas.DataFrame() for k in range(sensors_ID.id.size)]
    for k in range(sensors_ID.id.size):
        File_name = "D:/New folder/Bety Python/Data/Image_data/AVG Sensor "+str(sensors_ID.id[k])+" 482 MHz_ 700 MHz spectrum data.csv"
        if(os.path.isfile(File_name)):
            series[k]= pandas.read_csv(File_name, header=0)
            if sensors_ID.id[k]==62:
                series[k]['Index'] = pandas.to_datetime(series[k]['Index'], format='%m/%d/%Y %H:%M')
            else:
                series[k]['Index'] = pandas.to_datetime(series[k]['Index'], format='%Y-%m-%d %H:%M:%S')
            series[k] = series[k].set_index('Index')
        else:
            print("Mesaurment data of sensor "+str(sensors_ID.id[k])+" doesn't exist!!")
    return [sensors, series]
