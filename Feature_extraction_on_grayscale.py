# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:48:47 2020

@author: sajid
"""

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import gabor
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
import os

infected = os.listdir('C:/Users/sajid/Desktop/MALARIA/cell_images/Parasitized/') 
uninfected = os.listdir('C:/Users/sajid/Desktop/MALARIA/cell_images/Uninfected/')

data =[]
lebel=[]
dataframe=pd.DataFrame()

for i in infected[:]:
    try:
        img_arr = io.imread('C:/Users/sajid/Desktop/MALARIA/cell_images/Parasitized/'+i)
        img_arr=cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        feat_lbp = local_binary_pattern(img_arr,8,1,'uniform')
        lbp_hist,_ = np.histogram(feat_lbp,8)
        lbp_hist = np.array(lbp_hist,dtype=float)
        lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob**2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob,np.log2(lbp_prob)))
        gCoMat = greycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True) # Co-occurance matrix
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        energy = greycoprops(gCoMat, prop='energy')
        correlation = greycoprops(gCoMat, prop='correlation')
        gaborFilt_real,gaborFilt_imag = gabor(img_arr,frequency=0.6)
        gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
        gabor_hist,_ = np.histogram(gaborFilt,8)
        gabor_hist = np.array(gabor_hist,dtype=float)
        gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
        gabor_energy = np.sum(gabor_prob**2)
        gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
        datas=[lbp_energy,lbp_entropy,contrast.item(),dissimilarity.item(),homogeneity.item(),energy.item(),correlation.item(),gabor_energy,gabor_entropy,1]
        data.append(datas)
        
    except AttributeError:
        print("")

for i in uninfected[:]:
    try:
        img_arr = io.imread('C:/Users/sajid/Desktop/MALARIA/cell_images/Uninfected/'+i)
        img_arr=cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        feat_lbp = local_binary_pattern(img_arr,8,1,'uniform')
        lbp_hist,_ = np.histogram(feat_lbp,8)
        lbp_hist = np.array(lbp_hist,dtype=float)
        lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob**2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob,np.log2(lbp_prob)))
        gCoMat = greycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True) # Co-occurance matrix
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        energy = greycoprops(gCoMat, prop='energy')
        correlation = greycoprops(gCoMat, prop='correlation')
        gaborFilt_real,gaborFilt_imag = gabor(img_arr,frequency=0.6)
        gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
        gabor_hist,_ = np.histogram(gaborFilt,8)
        gabor_hist = np.array(gabor_hist,dtype=float)
        gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
        gabor_energy = np.sum(gabor_prob**2)
        gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
        datas=[lbp_energy,lbp_entropy,contrast.item(),dissimilarity.item(),homogeneity.item(),energy.item(),correlation.item(),gabor_energy,gabor_entropy,0]
        data.append(datas)
        
    except AttributeError:
        print("")


dataset= pd.DataFrame(data,columns=['lbp_energy','lbp_entropy','contrast','dissimilarity','homogeneity','energy','correlation','gabor_energy','gabor_entropy','class'])
dataset.to_csv('C:/Users/sajid/Desktop/MALARIA/DATASET2.csv')