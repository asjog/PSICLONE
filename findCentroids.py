import os
import numpy as np
import nibabel as nib
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn import mixture
from scipy.optimize import *
import sympy as sp




#img_file_name = 'data/KKI2009-01-MPRAGE.nii'
img_file_name = 'data/1000505_mprage.nii'
img = nib.load(img_file_name)
data = img.get_data()
dataf = data.flatten('F')
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
b = np.asarray([c1_mu, c2_mu, c3_mu])
bidxs = np.argsort(b)
sorted_gmm_labels = np.zeros((gmm_labels.size,1))
for c in range(0,3):
    sorted_gmm_labels[(gmm_labels==c).nonzero()] =  (bidxs==c).nonzero()

bsort = np.sort(b)
msc = bsort[0]
msg = bsort[1]
msw = bsort[2]
labelf = np.array(dataf)
labelf[nz_idxs] = sorted_gmm_labels.flatten() #ms.labels_
labeldata = np.reshape(labelf,img.shape, 'F')
labelimg = nib.Nifti1Image(labeldata,img.affine)
labelimg.set_data_dtype(np.int16)
labelimg.header.set_xyzt_units(img.header.get_xyzt_units()[0], img.header.get_xyzt_units()[1])
#labelimg.to_filename('data/KKI2009-01-MPRAGE_GMM.nii')
#nib.save(labelimg, 'data/KKI2009-01-MPRAGE_GMM.nii')
labelimg.to_filename('data/1000505_mprage_GMM.nii')
nib.save(labelimg, 'data/1000505_mprage_GMM.nii')
pdc = 1.09
pdg = 0.84
pdw = 0.72

t1c = 3688
t1g = 1308
t1w = 801

t2c = 1137
t2g = 117
t2w = 81

# def myFunction(z):
#     thetam = z[0]
#     TIm = z[1]
#     TD = z[2]
#     tau =  2600
#     F = np.empty((3))
#     F[0] = np.log(pdc) + thetam/100 + np.log(np.abs(1 - 2*np.exp(-TIm/t1c)/(1 + np.exp(-(TIm + TD + tau)/t1c)))) - np.log(msc)
#
#     F[1] = np.log(pdg) + thetam/100 + np.log(np.abs(1 - 2*np.exp(-TIm/t1g)/(1 + np.exp(-(TIm + TD + tau)/t1g)))) - np.log(msg)
#
#     F[2] = np.log(pdw) + thetam/100 + np.log(np.abs(1 - 2*np.exp(-TIm/t1w)/(1 + np.exp(-(TIm + TD + tau)/t1w)))) - np.log(msw)
#
#     return F




# log(pdc) + thetam + log(abs(1- 2 * exp(-TIm/t1c) / (1 + exp(-(TIm + TD + tau)/t1c)) )) = log(msc)
# zguess = np.array([800,1200,800])
# z = fsolve(myFunction,zguess)

thetam, TIm, TD  = sp.symbols('thetam, TIm, TD')
tau = 2600
efc = sp.log(pdc) + thetam + sp.log((1 - 2*sp.exp(-TIm/t1c)/(1 + sp.exp(-(TIm + TD + tau)/t1c)))) - sp.log(msc) - sp.log(100)
efg = sp.log(pdg) + thetam + sp.log((1 - 2*sp.exp(-TIm/t1g)/(1 + sp.exp(-(TIm + TD + tau)/t1g)))) - sp.log(msg)- sp.log(100)
efw = sp.log(pdw) + thetam + sp.log((1 - 2*sp.exp(-TIm/t1w)/(1 + sp.exp(-(TIm + TD + tau)/t1w)))) - sp.log(msw)- sp.log(100)
sol = sp.nsolve((efc, efg, efw),(thetam, TIm,TD), (800,1900,800))

thetamh = float(sol[0])
TImh = float(sol[1])
TDh = float(sol[2])

# read atlas betaspace
betaspace_filename = '/home/amod/mimecs/IPMI2013_BetaspaceRegression/code/msAtlas_hybrid_pd_t1_mprage_t2_true_betaspace.csv'
ms_atlas_betaspace = np.genfromtxt(betaspace_filename, delimiter=',')

atlas_pd = np.array(ms_atlas_betaspace[:,0].tolist(),dtype=np.float32)
atlas_t1 = np.array(ms_atlas_betaspace[:,1].tolist(),dtype=np.float32)
atlas_t2 = np.array(ms_atlas_betaspace[:,2].tolist(),dtype=np.float32)

synth_ms_atlas_fg = np.exp(np.log(atlas_pd) + thetamh + np.log(1 - 2*np.exp(-TImh/atlas_t1)/(1 + np.exp(-(TImh + TDh + tau)/atlas_t1)))- np.log(100))

synthf = np.array(dataf)
synthf[nz_idxs] = synth_ms_atlas_fg
synthdata = np.reshape(synthf,img.shape,'F')
synthimg = nib.Nifti1Image(synthdata,img.affine)
synthimg.set_data_dtype(np.float32)
synthimg.header.set_xyzt_units(img.header.get_xyzt_units()[0], img.header.get_xyzt_units()[1])
#labelimg.to_filename('data/KKI2009-01-MPRAGE_GMM.nii')
#nib.save(labelimg, 'data/KKI2009-01-MPRAGE_GMM.nii')
synthimg.to_filename('data/1000505_synth_mprage_nsolve_init_TI1900.nii')
nib.save(synthimg, 'data/1000505_synth_mprage_nsolve_init_TI1900.nii')