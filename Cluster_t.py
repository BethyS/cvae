## ## Bethelhem S.: cluster_temporal data script - Py
## ## (Version 1.0.1, built: 2020-10-01)
## ## Copyright (C)2020 Bethelhem Seifu

## import library -----------------------------------------------
import numpy
import pandas

import tslearn
from tslearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
import sklearn
import itertools
from sklearn import mixture, preprocessing,cluster

seed = 0
numpy.random.seed(seed)

## loading dataset -------------------------------------------
dataset = [];

deif
sensor_ID = [113, 196, 201, 230, 323, 358, 363, 378, 389, 395, 397, 62]
for k in range(12):
    series = pandas.read_csv("E:\Bety Python\Data\Image_data\\AVG Sensor "+str(sensor_ID[k-1])+
              " 482 MHz_ 700 MHz spectrum data.csv", header=0, index_col=None,low_memory=False
              )
    # if (k==0):
    #     series['Index'] = pandas.to_datetime(series['Index'], format='%m/%d/%Y %H:%M')
    #     series = series.set_index('Index')
    # else:
    #     series['Index'] = pandas.to_datetime(series['Index'], format='%Y-%m-%d %H:%M:%S')
    #     series = series.set_index('Index')
    dataset.append(series)

   
#raw data 
rawdata  = dataset[6]
rawdata['Index'] = pandas.to_datetime(rawdata['Index'], format='%Y-%m-%d %H:%M:%S')
rawdata.dtypes
rawdata = rawdata.set_index('Index')

dataset[2]['482 MHz'].plot(linewidth=0.5)


series2D = rawdata.to_numpy()[0:850,0:20]
series3D =numpy.array(numpy.vsplit(series2D,170))

index= pandas.MultiIndex.from_product([range(s) for s in series2D.shape])
series_df= pandas.DataFrame({'series2D':series2D.flatten()},index=index).reset_index()

#numpy.random.shuffle(series3D)
# K means----------------------------------------------


X_train = tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(series_df)

sz = X_train.shape[1]

inertia = []
max_iter = 10
gama = 0.0001
for k in range(15):
    model = tslearn.clustering.TimeSeriesKMeans(n_clusters=(k + 1), metric="softdtw", max_iter=max_iter,
                             metric_params={"gamma": gama}, verbose=True, random_state=seed)
    series2D_kmeans = model.fit(X_train)
    inertia.append([(k + 1), model.inertia_])

plt.plot(numpy.array(inertia)[:, 0], numpy.array(inertia)[:, 1], 'bo', numpy.array(inertia)[:, 0],
         numpy.array(inertia)[:, 1], 'k', lw=2)
plt.grid(True)
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia')
plt.title('Cluster Inertia plot')
plt.show()
print(series2D_kmeans.cluster_centers_.shape)

print("Euclidean k-means")
clus = 4
model = tslearn.clustering.TimeSeriesKMeans(n_clusters=clus, metric="softdtw", max_iter=max_iter,
                         metric_params={"gamma": gama}, verbose=True, random_state=seed)
y_pred = model.fit_predict(X_train)

# time-series plot

for yi in range(clus):
    plt.subplot(1, clus, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

#plt.tight_layout()
plt.show()

## GMM implementation-------------------------------------------------
X_train = series3D[:,:,0]
lowest_bic = numpy.infty
bic = []
n_components_range = range(1, 9)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type, 
                                      max_iter=150,init_params='kmeans')
        gmm.fit(X_train)
        bic.append(gmm.bic(X_train))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
bic = numpy.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []
# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = numpy.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = numpy.mod(bic.argmin(), len(n_components_range)) + .65 + \
       .2 * numpy.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X_train)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not numpy.any(Y_ == i):
        continue
    plt.scatter(X_train[Y_ == i, 0], X_train[Y_ == i, 1], .8, color=color)

    #Plot an ellipse to show the Gaussian component
    angle = numpy.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / numpy.pi  # convert to degrees
    ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()
