## ## Bethelhem S.: Main script - Py
## ## (Version 1.0.1, built: 2020-10-01)
## ## Copyright (C)2020 Bethelhem Seifu

#import library 
import os
import numpy 
import pandas
import datetime
import tslearn
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

# from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

#set working dir
os.chdir('E:\\Bety Python\\Data\\cvae')

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})

#callout dataset
seed = 0

datetime_object = datetime.datetime.strptime('2020-08-03 01:34:13 AM', '%Y-%m-%d %I:%M:%S %p')

series_time = pandas.read_csv('E:\Bety Python\Data\\raw2d.csv', header=0)
series['Index'] = pandas.to_datetime(series['Index'], format='%Y-%m-%d %H:%M:%S')
series.head(3)
series.dtypes
series = series.set_index('Index')

series.plot(linewidth=0.5)

plt.show()
model = tslearn.clustering.TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=10)
model.fit(series)
def kmean_cluster(X,Max_clus=15):
    inertia = []
    max_iter = 1000
    gama = 0.0001
    for k in range(15):
        model = tslearn.clustering.TimeSeriesKMeans(n_clusters=(k + 1), metric="softdtw", max_iter=max_iter,
                             metric_params={"gamma": gama}, verbose=True, random_state=seed)
        series3D_kmeans = model.fit(X)
        inertia.append([(k + 1), model.inertia_])
    return inertia,series3D_kmeans


    
