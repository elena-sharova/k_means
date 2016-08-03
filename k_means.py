# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 11:54:53 2016

@author: Elena
"""

import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def generate_data(mean, covar, num_points):
    # Generates data from a normal distribution

    data = np.random.multivariate_normal(mean, covar, num_points)
    return data
    
def predict_assignments(model, data):
    # GUse k-means to predict each data point assignment to k1 or k2

    x1=[], x2=[], y1=[], y2=[]
    res = k_means.predict(data)
    i=0
    for r in res:
        if r==0:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
        i+=1
    return x1, y1, x2, y2
    
if __name__=='__main__':
        
    # Simulate random points from two clusters        
    means_k1 = [0.6, 0.7]
    covar_k1 = [[0.2, 0],[0.02, 0.2]]
    
    means_k2 = [0.02, 0.12]
    covar_k2 = [[0.03, 0],[0.001, 0.03]]
    
    n=100
    np.random.seed(4)
    x_k1, y_k1 = generate_data(means_k1,covar_k1,n).T
    x_k2, y_k2 = generate_data(means_k2,covar_k2,n).T
    
    # Plot the data
    plt.subplot(121)
    plt.ylim(-1,2)
    plt.xlim(-1,2)
    plt.title("Actual")
    plt.plot(x_k1, y_k1, 'bo')
    plt.plot(x_k2, y_k2, 'ro')
    
    conc_x=np.concatenate((x_k1,x_k2))
    conc_y=np.concatenate((y_k1,y_k2))
    data=np.column_stack((conc_x, conc_y))
    
    # Use k-means to discover clusters
    k_means = KMeans(n_clusters=2)
    k_means.fit_predict(data)
    k_means_centers = k_means.cluster_centers_
    
    plt.plot(k_means_centers[0][0],k_means_centers[0][1], 'o',ms=11.0,c='black')
    plt.plot(k_means_centers[1][0],k_means_centers[1][1], 'o',ms=11.0,c='black')

    
    # Use the cluster centers to re-interprete the cluster assignments
    x1,y1,x2,y2 = predict_assignments(k_means, data)
    
    plt.subplot(122)
    plt.ylim(-1,2)
    plt.xlim(-1,2)
    plt.title("Predicted")
    plt.plot(x1,y1, 'bo')
    plt.plot(x2,y2, 'ro')
    plt.show()