import os
import numpy as np
import nibabel as nib
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn import mixture
img_file_name = 'data/KKI2009-01-MPRAGE.nii'
img = nib.load(img_file_name)
data = img.get_data()
dataf = data.flatten()
nz_idxs = np.where(dataf > 0)[0]
fg = dataf[dataf>0]
fg = fg.reshape(fg.__len__(),1)
bw = 150
#estimate_bandwidth(fg,quantile=0.2,n_samples=500)
#ms = MeanShift(bandwidth=bw,bin_seeding=True)
#ms.fit(fg)
clf = mixture.GMM(n_components=3, covariance_type='full')
gmm_labels = clf.fit_predict(fg)
c1_mu = np.mean(fg[gmm_labels==0])
c2_mu = np.mean(fg[gmm_labels==1])
c3_mu = np.mean(fg[gmm_labels==2])
b = [c1_mu, c2_mu, c3_mu]
bidxs = np.argsort(b)
sorted_gmm_labels = np.zeros((gmm_labels.size,1))
for c in range(0,3):
    sorted_gmm_labels[(gmm_labels==c).nonzero()] =  (bidxs==c).nonzero()


dataf[nz_idxs] = sorted_gmm_labels.flatten() #ms.labels_
labeldata = np.reshape(dataf,(256,256,95))
labelimg = nib.Nifti1Image(labeldata,img.affine)
labelimg.set_data_dtype(np.int16)
labelimg.header.set_xyzt_units(img.header.get_xyzt_units()[0], img.header.get_xyzt_units()[1])
labelimg.to_filename('data/KKI2009-01-MPRAGE_GMM.nii')
nib.save(labelimg, 'data/KKI2009-01-MPRAGE_GMM.nii')