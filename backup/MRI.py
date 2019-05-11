
############################
############################
## working with MR images ##
####### in ADNI data #######
############################
############################


from sklearn.tree import _tree
from sklearn import tree
import graphviz
from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import csv
import pandas as pd
from statistics import mean
from statistics import variance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from statistics import median
import numpy as np
import sys, argparse
import scipy
import matplotlib as plt
from csv import reader
import mglearn
import copy
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import os
import subprocess
from __future__ import division
import math
import os, sys
import subprocess
import sklearn
from sklearn.tree import _tree
# for MR images
from nipy import load_image
from nilearn import plotting
from nilearn import image
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF



def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


AD_002_S_0619__1 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR__GradWarp__N3__Scaled_Br_20070717184209073_S24022_I60451.nii')
AD_002_S_0619__2 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_2_Br_20081001115218896_S15145_I118678.nii')
AD_002_S_0619__3 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_Br_20070411125458928_S15145_I48617.nii')
AD_002_S_0619__4 = load_image(filename='0_AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_Br_20070816100717385_S33969_I67871.nii')
AD_002_S_0816__1 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081002102135862_S18402_I118984.nii')
AD_002_S_0816__2 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217005829488_S18402_I40731.nii')
AD_002_S_0816__3 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070717185335251_S29612_I60465.nii')
AD_002_S_0816__4 = load_image(filename='0_AD/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080224131120787_S45030_I92146.nii')
AD_002_S_0938__1 = load_image(filename='0_AD/ADNI_002_S_0938_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219175406282_S19852_I40980.nii')
AD_002_S_1018__1 = load_image(filename='0_AD/ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217030439623_S23128_I40817.nii')
AD_005_S_0221__1 = load_image(filename='0_AD/ADNI_005_S_0221_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080410142121502_S28459_I102054.nii')
AD_005_S_0221__2 = load_image(filename='0_AD/ADNI_005_S_0221_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20061212173139396_S19846_I32899.nii')
AD_007_S_0316__1 = load_image(filename='0_AD/ADNI_007_S_0316_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070118024327307_S21488_I36559.nii')
AD_007_S_0316__2 = load_image(filename='0_AD/ADNI_007_S_0316_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070923130250878_S31849_I74627.nii')
AD_007_S_1339__1 = load_image(filename='0_AD/ADNI_007_S_1339_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070607134807952_S27414_I56319.nii')
AD_007_S_1339__2 = load_image(filename='0_AD/ADNI_007_S_1339_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071027133744003_S41402_I78754.nii')
AD_007_S_1339__3 = load_image(filename='0_AD/ADNI_007_S_1339_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080430143918869_S49074_I104363.nii')
AD_010_S_0786__1 = load_image(filename='0_AD/ADNI_010_S_0786_MR_MPR____N3__Scaled_2_Br_20081002102855696_S19638_I118990.nii')
AD_010_S_0829__1 = load_image(filename='0_AD/ADNI_010_S_0829_MR_MPR____N3__Scaled_Br_20080410112243910_S46875_I101977.nii')
AD_011_S_0003__1 = load_image(filename='0_AD/ADNI_011_S_0003_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070109190839611_S19096_I35576.nii')
AD_011_S_0010__1 = load_image(filename='0_AD/ADNI_011_S_0010_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080229151344346_S45936_I94368.nii')
AD_011_S_0053__1 = load_image(filename='0_AD/ADNI_011_S_0053_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070810172103015_S23166_I66945.nii')
AD_067_S_0076__1 = load_image(filename='0_AD/ADNI_067_S_0076_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081003155853706_S16974_I119191.nii')
AD_067_S_0076__2 = load_image(filename='0_AD/ADNI_067_S_0076_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061229172001056_S16974_I34788.nii')
AD_094_S_1164__1 = load_image(filename='0_AD/ADNI_094_S_1164_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080220140841993_S44407_I91057.nii')
AD_094_S_1164__2 = load_image(filename='0_AD/ADNI_094_S_1164_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081016191406569_S51749_I121632.nii')
AD_094_S_1397__1 = load_image(filename='0_AD/ADNI_094_S_1397_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20120731130350494_S50305_I319849.nii')
AD_094_S_1397__2 = load_image(filename='0_AD/ADNI_094_S_1397_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080307104443256_S31011_I95662.nii')


NC_002_S_0295__1 = load_image(filename='0_NC/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120532722_S21856_I118692.nii')
NC_002_S_0295__2 = load_image(filename='0_NC/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii')
NC_002_S_0295__3 = load_image(filename='0_NC/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319113623975_S13408_I45108.nii')
NC_002_S_0413__1 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114742166_S13893_I118673.nii')
NC_002_S_0413__2 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120813046_S22557_I118695.nii')
NC_002_S_0413__3 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319115331858_S13893_I45117.nii')
NC_002_S_0413__4 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070713121420365_S32938_I60008.nii')
NC_002_S_0413__5 = load_image(filename='0_NC/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071028190559976_S22557_I79122.nii')
NC_002_S_0685__1 = load_image(filename='0_NC/ADNI_002_S_0685_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20071223122419058_S25369_I86020.nii')
NC_002_S_1261__1 = load_image(filename='0_NC/ADNI_002_S_1261_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080613103237200_S50898_I109394.nii')
NC_002_S_1280__1 = load_image(filename='0_NC/ADNI_002_S_1280_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071110105845543_S38235_I81321.nii')
NC_003_S_0981__1 = load_image(filename='0_NC/ADNI_003_S_0981_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080131105013455_S31564_I89046.nii')
NC_005_S_0223__1 = load_image(filename='0_NC/ADNI_005_S_0223_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080410141453831_S28246_I102045.nii')
NC_006_S_0731__1 = load_image(filename='0_NC/ADNI_006_S_0731_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070804130615519_S27980_I64575.nii')
NC_007_S_0068__1 = load_image(filename='0_NC/ADNI_007_S_0068_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070110214628818_S23109_I35773.nii')
NC_007_S_0070__1 = load_image(filename='0_NC/ADNI_007_S_0070_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070118014059657_S18818_I36523.nii')
NC_007_S_1206__1 = load_image(filename='0_NC/ADNI_007_S_1206_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070923131606579_S38141_I74645.nii')
NC_010_S_0067__1 = load_image(filename='0_NC/ADNI_010_S_0067_MR_MPR____N3__Scaled_2_Br_20081001122414391_S25341_I118706.nii')
NC_010_S_0419__1 = load_image(filename='0_NC/ADNI_010_S_0419_MR_MPR____N3__Scaled_Br_20070731161135546_S24112_I63325.nii')
NC_011_S_0005__1 = load_image(filename='0_NC/ADNI_011_S_0005_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061206162204955_S12037_I31885.nii')
NC_011_S_0021__1 = load_image(filename='0_NC/ADNI_011_S_0021_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061206174058589_S22065_I31970.nii')
NC_011_S_0023__1 = load_image(filename='0_NC/ADNI_011_S_0023_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070808160104173_S23153_I65902.nii')
NC_067_S_0056__1 = load_image(filename='0_NC/ADNI_067_S_0056_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081003155535571_S8723_I119186.nii')
NC_067_S_0056__2 = load_image(filename='0_NC/ADNI_067_S_0056_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070209152131877_S16358_I38678.nii')
NC_094_S_1241__1 = load_image(filename='0_NC/ADNI_094_S_1241_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081016163336876_S50440_I121470.nii')
NC_094_S_1241__2 = load_image(filename='0_NC/ADNI_094_S_1241_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20081016181908278_S38195_I121579.nii')

# create lists of data and names
list_names_AD = ['AD_002_S_0619__1', 'AD_002_S_0619__2', 'AD_002_S_0619__3', 'AD_002_S_0619__4', 'AD_002_S_0816__1', 'AD_002_S_0816__2', 'AD_002_S_0816__3', 'AD_002_S_0816__4', 'AD_002_S_0938__1', 'AD_002_S_1018__1', 'AD_005_S_0221__1', 'AD_005_S_0221__2', 'AD_007_S_0316__1', 'AD_007_S_0316__2', 'AD_007_S_1339__1', 'AD_007_S_1339__2', 'AD_007_S_1339__3', 'AD_010_S_0786__1', 'AD_010_S_0829__1', 'AD_011_S_0003__1', 'AD_011_S_0010__1', 'AD_011_S_0053__1', 'AD_067_S_0076__1', 'AD_067_S_0076__2', 'AD_094_S_1164__1', 'AD_094_S_1164__2', 'AD_094_S_1397__1', 'AD_094_S_1397__2']
list_AD = [AD_002_S_0619__1, AD_002_S_0619__2, AD_002_S_0619__3, AD_002_S_0619__4, AD_002_S_0816__1, AD_002_S_0816__2, AD_002_S_0816__3, AD_002_S_0816__4, AD_002_S_0938__1, AD_002_S_1018__1, AD_005_S_0221__1, AD_005_S_0221__2, AD_007_S_0316__1, AD_007_S_0316__2, AD_007_S_1339__1, AD_007_S_1339__2, AD_007_S_1339__3, AD_010_S_0786__1, AD_010_S_0829__1, AD_011_S_0003__1, AD_011_S_0010__1, AD_011_S_0053__1, AD_067_S_0076__1, AD_067_S_0076__2, AD_094_S_1164__1, AD_094_S_1164__2, AD_094_S_1397__1, AD_094_S_1397__2]
list_names_NC = ['NC_002_S_0295__1', 'NC_002_S_0295__2', 'NC_002_S_0295__3', 'NC_002_S_0413__1', 'NC_002_S_0413__2', 'NC_002_S_0413__3', 'NC_002_S_0413__4', 'NC_002_S_0413__5', 'NC_002_S_0685__1', 'NC_002_S_1261__1', 'NC_002_S_1280__1', 'NC_003_S_0981__1', 'NC_005_S_0223__1', 'NC_006_S_0731__1', 'NC_007_S_0068__1', 'NC_007_S_0070__1', 'NC_007_S_1206__1', 'NC_010_S_0067__1', 'NC_010_S_0419__1', 'NC_011_S_0005__1', 'NC_011_S_0021__1', 'NC_011_S_0023__1', 'NC_067_S_0056__1', 'NC_067_S_0056__2', 'NC_094_S_1241__1', 'NC_094_S_1241__2']
list_NC= [NC_002_S_0295__1, NC_002_S_0295__2, NC_002_S_0295__3, NC_002_S_0413__1, NC_002_S_0413__2, NC_002_S_0413__3, NC_002_S_0413__4, NC_002_S_0413__5, NC_002_S_0685__1, NC_002_S_1261__1, NC_002_S_1280__1, NC_003_S_0981__1, NC_005_S_0223__1, NC_006_S_0731__1, NC_007_S_0068__1, NC_007_S_0070__1, NC_007_S_1206__1, NC_010_S_0067__1, NC_010_S_0419__1, NC_011_S_0005__1, NC_011_S_0021__1, NC_011_S_0023__1, NC_067_S_0056__1, NC_067_S_0056__2, NC_094_S_1241__1, NC_094_S_1241__2]

# get array of MR image
def get_data(names):
    l = []
    for n in names:
        l.append(n.get_data())
    
    return l

def get_sumsumsumdata(obj):
    l=[]
    for o in obj:
        l.append(sum(sum(sum(o))))
    
    return l

def get_mean_data(liste):
    l=[]
    
    for o in liste:
        l.append(np.nanmean(o))
    
    return l

def get_maxsumsum_data(liste):
    l=[]
    for o in liste:
        l.append(max(sum(sum(o))))
    return l

def get_slice0(liste):
    l = []
    
    for o in liste:
        fst = o.shape[0]

        l.append(o[(fst//2), :, :])
    
    return l

def get_slice1(liste):
    l=[]
    for o in liste:
        snd = o.shape[1]
        l.append(o[:, snd//2, :])
    return l

def get_slice2(liste):
    l=[]
    for o in liste:
        trd = o.shape[2]
        l.append(o[:, :, trd//2])
    return l

def get_sumsumsslice(liste):
    l=[]
    for o in liste:
        l.append(sum(sum(o)))
    return l

def get_meanslice(liste):
    l=[]
    for o in liste:
        l.append(np.nanmean(o))
    return l

def get_maxsumslice(liste):
    l =[]
    for o in liste:
        l.append(max(sum(o)))
                 
    return l

def get_meansumslice(liste):
    l=[]
    for o in liste:
        l.append(np.nanmean(sum(o)))
    return l

def get_maxsum(liste):
    l = []
    
    for o in liste:
        l.append(max(o))
        
    return l

def get_flat_list(liste):
    flat_list =[]
    for sublist in liste:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def get_maxflat(liste):
    
    l = []
    for o in liste:
        list_flat = get_flat_list(o)
        l.append(max(list_flat))
        
    return l





# get probabilites of tissues in brain for CN (normal control group)
def get_tissue_sums_CN_smooth(list_data):
    global white_CN_smooth
    global gray_CN_smooth
    global csf_CN_smooth
    global white_sum_CN_smooth
    global gray_sum_CN_smooth
    global csf_sum_CN_smooth
    
    i=0
    for o in list_data:
        try:
            hmrf = TissueClassifierHMRF()
            initial_segmentation, final_segmentation, PVE = hmrf.classify(o, nclass, beta)

            img_ax = np.rot90(final_segmentation[..., 89])
            img_cor = np.rot90(final_segmentation[:, 128, :])

            img_ax = np.rot90(PVE[..., 89, 0])
            #CSF
            csf_CN_smooth.append(img_ax)
            csf_sum_CN_smooth.append(sum(img_ax))

            img_cor = np.rot90(PVE[:, :, 89, 1])
            #GRAY
            gray_CN_smooth.append(img_cor)
            gray_sum_CN_smooth.append(sum(img_cor))

            img_cor = np.rot90(PVE[:, :, 89, 2])
            #WHITE
            white_CN_smooth.append(img_cor)
            white_sum_CN_smooth.append(sum(img_cor))
            i=i+1
        except ValueError as error:
            i=i+1
            print error
            print i



# exemplary usage of functions above to extract features from MR images
def create_features():
	list_data_AD = get_data(list_AD)
	list_data_NC = get_data(list_NC)
	list_sumsumsumAD = get_sumsumsumdata(list_data_AD)


	x = sum(sum(AD_002_S_0619__3.get_data()))
	y = sum(sum(AD_002_S_0816__2.get_data()))
	x[np.isnan(x)]=0
	y[np.isnan(y)]=0
	y[np.isinf(y)]=0
	list_sumsumsumAD[2] = sum(x)
	list_sumsumsumAD[5] = sum(y)

	list_sumsumsumNC = get_sumsumsumdata(list_data_NC)
	z = sum(sum(NC_002_S_0295__2.get_data()))
	z[np.isnan(z)]=0
	list_sumsumsumNC[1] = sum(z)

	list_mean_AD = get_mean_data(list_data_AD)

	list_mean_AD[5] = np.nanmean(list_mean_AD)

	list_mean_NC = get_mean_data(list_data_NC)
	list_mean_NC[1] = np.nanmean(list_mean_NC)

	list_maxsumsum_AD = get_maxsumsum_data(list_data_AD)
	del list_maxsumsum_AD[5]
	tmp = np.mean(list_maxsumsum_AD)
	list_maxsumsum_AD = get_maxsumsum_data(list_data_AD)
	list_maxsumsum_AD[5] = tmp

	list_maxsumsum_NC = get_maxsumsum_data(list_data_NC)

	list_slice0_AD = get_slice0(list_data_AD)
	list_slice0_NC = get_slice0(list_data_NC)

	list_slice1_AD = get_slice1(list_data_AD)
	list_slice1_NC = get_slice1(list_data_NC)

	list_slice2_AD = get_slice2(list_data_AD)
	list_slice2_NC = get_slice2(list_data_NC)

	list_sumslice0_AD = get_sumsumsslice(list_slice0_AD)
	list_sumslice0_NC = get_sumsumsslice(list_slice0_NC)
	list_sumslice0_NC[1] = np.nanmean(list_sumslice0_NC)

	list_sumslice1_AD = get_sumsumsslice(list_slice1_AD)
	list_sumslice1_NC = get_sumsumsslice(list_slice1_NC)
	list_sumslice1_NC[1] = np.nanmean(list_sumslice0_NC)

	list_sumslice2_AD = get_sumsumsslice(list_slice2_AD)
	list_sumslice2_NC = get_sumsumsslice(list_slice2_NC)

	list_meanslice0_AD = get_meanslice(list_slice0_AD)
	list_meanslice0_NC = get_meanslice(list_slice0_NC)
	list_meanslice1_AD = get_meanslice(list_slice1_AD)
	list_meanslice1_NC = get_meanslice(list_slice1_NC)
	list_meanslice2_AD = get_meanslice(list_slice2_AD)
	list_meanslice2_NC = get_meanslice(list_slice2_NC)

	list_maxsumslice0_AD = get_maxsumslice(list_slice0_AD)
	list_maxsumslice0_NC = get_maxsumslice(list_slice0_NC)
	del list_maxsumslice0_NC[1]
	tmp2 = np.mean(list_maxsumslice0_NC)
	list_maxsumslice0_NC[1] = tmp2

	list_maxsumslice1_AD = get_maxsumslice(list_slice1_AD)
	list_maxsumslice1_NC = get_maxsumslice(list_slice1_NC)
	list_maxsumslice2_AD = get_maxsumslice(list_slice2_AD)
	list_maxsumslice2_NC = get_maxsumslice(list_slice2_NC)


	list_meansumslice0_AD = get_meansumslice(list_slice0_AD)
	list_meansumslice0_NC = get_meansumslice(list_slice0_NC)
	del list_meansumslice0_NC[1]
	tmp3 = np.nanmean(list_meansumslice0_NC)
	list_meansumslice0_NC[1] = tmp3
	list_meansumslice1_AD = get_meansumslice(list_slice1_AD)
	list_meansumslice1_NC = get_meansumslice(list_slice1_NC)
	del list_meansumslice1_NC[1]
	tmp4 = np.mean(list_meansumslice1_NC)
	list_meansumslice1_NC[1] = tmp4
	list_mean_sumslice2_AD = get_meansumslice(list_slice2_AD)
	list_mean_sumslice2_NC = get_meansumslice(list_slice2_NC)

def get_lists_of_MR_data():
	list_names_AD_smoothed2 = ['AD_005_S_0221__3_smoothed', 'AD_005_S_0814__1_smoothed', 'AD_005_S_1341__1_smoothed', 'AD_005_S_1341__2_smoothed', 'AD_067_S_0076__3_smoothed', 'AD_068_S_0109__1_smoothed', 'AD_068_S_0109__2_smoothed', 'AD_082_S_1377__1_smoothed', 'AD_082_S_1377__2_smoothed', 'AD_094_S_1090__1_smoothed', 'AD_094_S_1090__2_smoothed', 'AD_094_S_1397__3_smoothed']
	list_names_NC_smoothed2 = ['NC_002_S_0685__2_smoothed', 'NC_002_S_0685__3_smoothed', 'NC_002_S_1261__2_smoothed', 'NC_002_S_1280__2_smoothed', 'NC_003_S_0907__1_smoothed', 'NC_005_S_0223__2_smoothed', 'NC_005_S_0553__1_smoothed', 'NC_068_S_0127__1_smoothed', 'NC_068_S_0127__2_smoothed', 'NC_068_S_0210__1_smoothed', 'NC_094_S_0692__1_smoothed']
	list_AD_smoothed2 = [AD_005_S_0221__3_smoothed, AD_005_S_0814__1_smoothed, AD_005_S_1341__1_smoothed, AD_005_S_1341__2_smoothed, AD_067_S_0076__3_smoothed, AD_068_S_0109__1_smoothed, AD_068_S_0109__2_smoothed, AD_082_S_1377__1_smoothed, AD_082_S_1377__2_smoothed, AD_094_S_1090__1_smoothed, AD_094_S_1090__2_smoothed, AD_094_S_1397__3_smoothed]
	list_NC_smoothed2 = [NC_002_S_0685__2_smoothed, NC_002_S_0685__3_smoothed, NC_002_S_1261__2_smoothed, NC_002_S_1280__2_smoothed, NC_003_S_0907__1_smoothed, NC_005_S_0223__2_smoothed, NC_005_S_0553__1_smoothed, NC_068_S_0127__1_smoothed, NC_068_S_0127__2_smoothed, NC_068_S_0210__1_smoothed, NC_094_S_0692__1_smoothed]
	list_AD2 = [AD_005_S_0221__3, AD_005_S_0814__1, AD_005_S_1341__1, AD_005_S_1341__2, AD_067_S_0076__3, AD_068_S_0109__1, AD_068_S_0109__2, AD_082_S_1377__1, AD_082_S_1377__2, AD_094_S_1090__1, AD_094_S_1090__2, AD_094_S_1397__3 ]
	list_NC2 = [NC_002_S_0685__2, NC_002_S_0685__3, NC_002_S_1261__2, NC_002_S_1280__2, NC_003_S_0907__1, NC_005_S_0223__2, NC_005_S_0553__1, NC_068_S_0127__1, NC_068_S_0127__2, NC_068_S_0210__1, NC_094_S_0692__1]
	list_names_AD2 = ['AD_005_S_0221__3', 'AD_005_S_0814__1', 'AD_005_S_1341__1', 'AD_005_S_1341__2', 'AD_067_S_0076__3', 'AD_068_S_0109__1', 'AD_068_S_0109__2', 'AD_082_S_1377__1', 'AD_082_S_1377__2', 'AD_094_S_1090__1', 'AD_094_S_1090__2', 'AD_094_S_1397__3']
	list_names_NC2 = ['NC_002_S_0685__2', 'NC_002_S_0685__3', 'NC_002_S_1261__2', 'NC_002_S_1280__2', 'NC_003_S_0907__1', 'NC_005_S_0223__2', 'NC_005_S_0553__1', 'NC_068_S_0127__1', 'NC_068_S_0127__2', 'NC_068_S_0210__1', 'NC_094_S_0692__1']
	list_names_AD_smoothed = ['AD_002_S_0619__1_smoothed', 'AD_002_S_0619__2_smoothed', 'AD_002_S_0619__3_smoothed', 'AD_002_S_0619__4_smoothed', 'AD_002_S_0816__1_smoothed', 'AD_002_S_0816__2_smoothed', 'AD_002_S_0816__3_smoothed', 'AD_002_S_0816__4_smoothed', 'AD_002_S_0938__1_smoothed', 'AD_002_S_1018__1_smoothed', 'AD_005_S_0221__1_smoothed', 'AD_005_S_0221__2_smoothed', 'AD_007_S_0316__1_smoothed', 'AD_007_S_0316__2_smoothed', 'AD_007_S_1339__1_smoothed', 'AD_007_S_1339__2_smoothed', 'AD_007_S_1339__3_smoothed', 'AD_010_S_0786__1_smoothed', 'AD_010_S_0829__1_smoothed', 'AD_011_S_0003__1_smoothed', 'AD_011_S_0010__1_smoothed', 'AD_011_S_0053__1_smoothed', 'AD_067_S_0076__1_smoothed', 'AD_067_S_0076__2_smoothed', 'AD_094_S_1164__1_smoothed', 'AD_094_S_1164__2_smoothed', 'AD_094_S_1397__1_smoothed', 'AD_094_S_1397__2_smoothed']
	list_AD_smoothed = [AD_002_S_0619__1_smoothed, AD_002_S_0619__2_smoothed, AD_002_S_0619__3_smoothed, AD_002_S_0619__4_smoothed, AD_002_S_0816__1_smoothed, AD_002_S_0816__2_smoothed, AD_002_S_0816__3_smoothed, AD_002_S_0816__4_smoothed, AD_002_S_0938__1_smoothed, AD_002_S_1018__1_smoothed, AD_005_S_0221__1_smoothed, AD_005_S_0221__2_smoothed, AD_007_S_0316__1_smoothed, AD_007_S_0316__2_smoothed, AD_007_S_1339__1_smoothed, AD_007_S_1339__2_smoothed, AD_007_S_1339__3_smoothed, AD_010_S_0786__1_smoothed, AD_010_S_0829__1_smoothed, AD_011_S_0003__1_smoothed, AD_011_S_0010__1_smoothed, AD_011_S_0053__1_smoothed, AD_067_S_0076__1_smoothed, AD_067_S_0076__2_smoothed, AD_094_S_1164__1_smoothed, AD_094_S_1164__2_smoothed, AD_094_S_1397__1_smoothed, AD_094_S_1397__2_smoothed]
	list_names_NC_smoothed = ['NC_002_S_0295__1_smoothed', 'NC_002_S_0295__2_smoothed', 'NC_002_S_0295__3_smoothed', 'NC_002_S_0413__1_smoothed', 'NC_002_S_0413__2_smoothed', 'NC_002_S_0413__3_smoothed', 'NC_002_S_0413__4_smoothed', 'NC_002_S_0413__5_smoothed', 'NC_002_S_0685__1_smoothed', 'NC_002_S_1261__1_smoothed', 'NC_002_S_1280__1_smoothed', 'NC_003_S_0981__1_smoothed', 'NC_005_S_0223__1_smoothed', 'NC_006_S_0731__1_smoothed', 'NC_007_S_0068__1_smoothed', 'NC_007_S_0070__1_smoothed', 'NC_007_S_1206__1_smoothed', 'NC_010_S_0067__1_smoothed', 'NC_010_S_0419__1_smoothed', 'NC_011_S_0005__1_smoothed', 'NC_011_S_0021__1_smoothed', 'NC_011_S_0023__1_smoothed', 'NC_067_S_0056__1_smoothed', 'NC_067_S_0056__2_smoothed', 'NC_094_S_1241__1_smoothed', 'NC_094_S_1241__2_smoothed']
	list_NC_smoothed = [NC_002_S_0295__1_smoothed, NC_002_S_0295__2_smoothed, NC_002_S_0295__3_smoothed, NC_002_S_0413__1_smoothed, NC_002_S_0413__2_smoothed, NC_002_S_0413__3_smoothed, NC_002_S_0413__4_smoothed, NC_002_S_0413__5_smoothed, NC_002_S_0685__1_smoothed, NC_002_S_1261__1_smoothed, NC_002_S_1280__1_smoothed, NC_003_S_0981__1_smoothed, NC_005_S_0223__1_smoothed, NC_006_S_0731__1_smoothed, NC_007_S_0068__1_smoothed, NC_007_S_0070__1_smoothed, NC_007_S_1206__1_smoothed, NC_010_S_0067__1_smoothed, NC_010_S_0419__1_smoothed, NC_011_S_0005__1_smoothed, NC_011_S_0021__1_smoothed, NC_011_S_0023__1_smoothed, NC_067_S_0056__1_smoothed, NC_067_S_0056__2_smoothed, NC_094_S_1241__1_smoothed, NC_094_S_1241__2_smoothed]
	list_names_AD = ['AD_002_S_0619__1', 'AD_002_S_0619__2', 'AD_002_S_0619__3', 'AD_002_S_0619__4', 'AD_002_S_0816__1', 'AD_002_S_0816__2', 'AD_002_S_0816__3', 'AD_002_S_0816__4', 'AD_002_S_0938__1', 'AD_002_S_1018__1', 'AD_005_S_0221__1', 'AD_005_S_0221__2', 'AD_007_S_0316__1', 'AD_007_S_0316__2', 'AD_007_S_1339__1', 'AD_007_S_1339__2', 'AD_007_S_1339__3', 'AD_010_S_0786__1', 'AD_010_S_0829__1', 'AD_011_S_0003__1', 'AD_011_S_0010__1', 'AD_011_S_0053__1', 'AD_067_S_0076__1', 'AD_067_S_0076__2', 'AD_094_S_1164__1', 'AD_094_S_1164__2', 'AD_094_S_1397__1', 'AD_094_S_1397__2']
	list_AD = [AD_002_S_0619__1, AD_002_S_0619__2, AD_002_S_0619__3, AD_002_S_0619__4, AD_002_S_0816__1, AD_002_S_0816__2, AD_002_S_0816__3, AD_002_S_0816__4, AD_002_S_0938__1, AD_002_S_1018__1, AD_005_S_0221__1, AD_005_S_0221__2, AD_007_S_0316__1, AD_007_S_0316__2, AD_007_S_1339__1, AD_007_S_1339__2, AD_007_S_1339__3, AD_010_S_0786__1, AD_010_S_0829__1, AD_011_S_0003__1, AD_011_S_0010__1, AD_011_S_0053__1, AD_067_S_0076__1, AD_067_S_0076__2, AD_094_S_1164__1, AD_094_S_1164__2, AD_094_S_1397__1, AD_094_S_1397__2]
	list_names_NC = ['NC_002_S_0295__1', 'NC_002_S_0295__2', 'NC_002_S_0295__3', 'NC_002_S_0413__1', 'NC_002_S_0413__2', 'NC_002_S_0413__3', 'NC_002_S_0413__4', 'NC_002_S_0413__5', 'NC_002_S_0685__1', 'NC_002_S_1261__1', 'NC_002_S_1280__1', 'NC_003_S_0981__1', 'NC_005_S_0223__1', 'NC_006_S_0731__1', 'NC_007_S_0068__1', 'NC_007_S_0070__1', 'NC_007_S_1206__1', 'NC_010_S_0067__1', 'NC_010_S_0419__1', 'NC_011_S_0005__1', 'NC_011_S_0021__1', 'NC_011_S_0023__1', 'NC_067_S_0056__1', 'NC_067_S_0056__2', 'NC_094_S_1241__1', 'NC_094_S_1241__2']
	list_NC= [NC_002_S_0295__1, NC_002_S_0295__2, NC_002_S_0295__3, NC_002_S_0413__1, NC_002_S_0413__2, NC_002_S_0413__3, NC_002_S_0413__4, NC_002_S_0413__5, NC_002_S_0685__1, NC_002_S_1261__1, NC_002_S_1280__1, NC_003_S_0981__1, NC_005_S_0223__1, NC_006_S_0731__1, NC_007_S_0068__1, NC_007_S_0070__1, NC_007_S_1206__1, NC_010_S_0067__1, NC_010_S_0419__1, NC_011_S_0005__1, NC_011_S_0021__1, NC_011_S_0023__1, NC_067_S_0056__1, NC_067_S_0056__2, NC_094_S_1241__1, NC_094_S_1241__2]

# exemplary creation of data frame from extracted features
def create_dataframe_from_features():
	df_whitesumsum_AD2 = pd.DataFrame()
	df_whitesumsum_AD2['whitesumsumAD2'] = get_sumsum(white_sum_AD2)
	df_whitesumsum_AD2.to_csv("whitesumsum_AD2.csv")

	df_whitesumsum_NC2 = pd.DataFrame()
	df_whitesumsum_NC2['whitesumsumNC2'] = get_sumsum(white_sum_CN2)
	df_whitesumsum_NC2.to_csv("whitesumsum_NC2.csv")

	df_graysumsum_AD2 = pd.DataFrame()
	df_graysumsum_AD2["graysumsum_AD2"] = get_sumsum(gray__sum_AD2)
	df_graysumsum_AD2.to_csv("graysumsum_AD2.csv")

	df_graysumsum_NC2 = pd.DataFrame()
	df_graysumsum_NC2["graysumsum_NC2"] = get_sumsum(gray_sum_CN2)
	df_graysumsum_NC2.to_csv("graysumsum_NC2.csv")

	df_csfsumsum_AD2 = pd.DataFrame()
	df_csfsumsum_AD2['csvsumsumAD2'] = get_sumsum(csf_sum_AD2)
	df_csfsumsum_AD2.to_csv("csfsumsum_AD2.csv")

	df_csfsumsum_NC2 = pd.DataFrame()
	df_csfsumsum_NC2['csfsumsumNC2'] = get_sumsum(csf_sum_CN2)
	df_csfsumsum_NC2.to_csv('csfsumsum_NC2.csv')

