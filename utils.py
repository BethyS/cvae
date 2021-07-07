# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:28:11 2021

@author: Bethe
"""
import numpy 
import pandas


import tslearn
from tslearn import preprocessing, clustering
from sklearn.mixture import GaussianMixture 

def load_dataset(sensors_ID=None):
    if(sensors_ID is None):
        sensors_ID= pandas.DataFrame({"id":[113, 196, 201, 230, 323, 358, 363, 378, 389, 395, 397, 62]})
    series=  [pandas.DataFrame() for k in range(sensors_ID.size)]
    for k in range(sensors_ID.size):
        File_name = "D:/New folder/Bety Python/Data/Image_data/AVG Sensor "+str(sensors_ID.id[k])+" 482 MHz_ 700 MHz spectrum data.csv"
        series[k]= pandas.read_csv(File_name, header=0)
        if sensors_ID.id[k]==62:
            series[k]['Index'] = pandas.to_datetime(series[k]['Index'], format='%m/%d/%Y %H:%M')
        else:
            series[k]['Index'] = pandas.to_datetime(series[k]['Index'], format='%Y-%m-%d %H:%M:%S')
        series[k] = series[k].set_index('Index')
    return series


def load_sensor_loc(Sensor_ID):
        sensors = pandas.read_csv("D:/New folder/Bety Python/Data/Sensor_List_new.csv", 
                                 header=0, index_col=None,low_memory=False)
        Sensor_loc = sensors[sensors.id.isin(sensors_ID.id)].iloc[:,[0,9,10,11,12]]
        return Sensor_loc

#clustering ..................

#in time dim ()

def kmean_cluster_time(X,Max_clus=15):
     kmeans_kwargs = {"init": "random", "n_init": 10,"max_iter": 1000,"random_state": 113,"metric":"softdtw","verbose":True}
     inertia = []
     for k in range(Max_clus):
        model = tslearn.clustering.TimeSeriesKMeans(n_clusters=(k + 1),metric_params={"gamma": 0.0001},**kmeans_kwargs )
        series2D_kmeans = model.fit(X)
        inertia.append([(k + 1), model.inertia_]) 
     return inertia,series2D_kmeans
    
def kmean_cluster(X,Max_clus=15):
     kmeans_kwargs = {"init": "random", "n_init": 10,"max_iter": 1000,"random_state": 113,"metric":"softdtw","verbose":True}
     inertia = []
     for k in range(Max_clus):
        model = tslearn.clustering.TimeSeriesKMeans(n_clusters=(k + 1),metric_params={"gamma": 0.0001},**kmeans_kwargs )
        series2D_kmeans = model.fit(X)
        inertia.append([(k + 1), model.inertia_]) 
     return inertia,series2D_kmeans
#Data Preprocessing 
def Data_preprocessing (dataset):
    na_index, _ = numpy.where(dataset.isna())
    if len(na_index)>0:
        end = na_index[0]// 10 * 10
        data = dataset.to_numpy()[0:end,:]
    else:
        data = dataset.to_numpy()
    index= pandas.MultiIndex.from_product([range(s) for s in data.shape])
    series_df= pandas.DataFrame({'data':data.flatten()},index=index).reset_index()
    X = tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(series_df)
    return X

def gmm_cluster(data,n_classes):
    lowest_bic = numpy.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in range(1, n_classes):
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type, 
                                      max_iter=150,init_params='kmeans')
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    bic = numpy.array(bic)
    return best_gmm

##  ------------------------------------------------
sensors_ID = pandas.DataFrame({"id":[201,363,389]})
Dataset= load_dataset(sensors_ID) 
sensor_loc= load_sensor_loc(sensors_ID) 
X=Data_preprocessing(Dataset[0])
gmm = gmm_cluster(X,8)
#gmm = gmm.fit(data)
