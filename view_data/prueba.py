import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math, os, sys
import numpy as np
import pandas as pd
import re
import scipy as sc
from scipy import stats


import scipy as sc
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.externals import joblib


#Se define el tamanio del voxel
voxel_size = 0.1
eps = np.finfo(np.float32).eps

#Se define la funcion para ubicar cada voxel dentro de la nube de puntos
def get_geom_feat( df):
    #------ XYZ -----
    X = np.array(df.X)
    Y = np.array(df.Y)
    Z = np.array(df.Z)
    X = X.T #- mean_x
    Y = Y.T #- mean_y
    Z = Z.T #- mean_z
    cov_mat = np.cov([X,Y,Z])
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    eig_val = np.sort(eig_val_cov)
    e1 = eig_val[2]
    e2 = eig_val[1]
    e3 = eig_val[0]
    #----- RGB -----
    R = np.array(df.R)
    G = np.array(df.G)
    B = np.array(df.B)
    R2= R-G-B
    R2= np.median(R)
    G2= G-R-B
    G2= np.mean(G)
    B2= B-R-G
    B2= np.mean(B)
    # Se calculan los descriptores de linealidad, planialidad, esfericidad, omnivarianza, anistropia, eigenentropia, sumatoria y cambio de curva
    if(e1>0 and e2>0 and e3>0):
        p0 = (e1-e2)/(e1 + eps)  # Linearity
        p1 = (e2-e3)/(e1 + eps)  # Planarity
        p2 =  e3/(e1 + eps)      # Sphericity
        p3 = pow(e1*e3*e3,1/3.0) # Omnivariance
        p4 = (e1-e3)/(e1 + eps)  # Anisotropy
        p5 = -( e3*np.log(e3) + e2*np.log(e2) + e1*np.log(e1) ) #Eigenentropy
        p6 = e1 +e3 + e3         # sumatory
        p7 = e3/(e1+e2+e3 + eps) #  change of curvature
    else:
        p0 = 0
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = 0
    # Calculo de angulos phi y theta
    mp = np.argmin(eig_val_cov)
    nx, ny, nz  = eig_vec_cov[:,mp] # https://mediatum.ub.tum.de/doc/800632/800632.pdf
    phi =abs(np.arctan(nz/ny))
    theta =abs(np.arctan((pow(nz,2)+pow(ny,2))/nx))
    return np.array([p0, p1, p2, p3, p4, p5, p6, p7, phi, theta,R2, G2, B2 ])

def features_calc(workfile, btfile):
    Xvox=[]
    Yvox=[]
    Zvox=[]
    #Abre el archivo y separa las columnas con sus respectios datos
    data = pd.read_csv( workfile, sep='\t', names = ["X","Y","Z","R","G","B"], header=2)
    X = data.X
    Y = data.Y
    Z = data.Z
    #Se crean las columnas del tamanio de los datos, rellenados de ceros
    data['D1']=np.zeros([len(X),1])
    data['D2']=np.zeros([len(X),1])
    data['D3']=np.zeros([len(X),1])
    data['D4']=np.zeros([len(X),1])
    data['D5']=np.zeros([len(X),1])
    data['D6']=np.zeros([len(X),1])
    data['D7']=np.zeros([len(X),1])
    data['D8']=np.zeros([len(X),1])
    data['D9']=np.zeros([len(X),1])
    data['D10']=np.zeros([len(X),1])
    data['D11']=np.zeros([len(X),1])
    data['D12']=np.zeros([len(X),1])
    data['D13']=np.zeros([len(X),1])
    wrfile = open(btfile,"r")
    translation = re.findall(r'Transform { translation (.*)', wrfile.read())
    L = len(translation)
    for k in range(0,L-1):
        a = translation[k]
        x,y,z,s = a.split(' ')
        Xvox.append(float(x))
        Yvox.append(float(y))
        Zvox.append(float(z))
    #Figura 1 centro de los voxeles
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(Xvox,Yvox,Zvox,s=10, c=Zvox)
    XYZ=np.array([X,Y,Z])
    XYZ=XYZ.T
    s = voxel_size
    L = len(Xvox)
    samples = 0
    POINTS_N= []
    i = 0    
    for k in range (0,L-1 ):
        #Se crean las regiones a partir del centro del voxel
        xmin=Xvox[k]-s/2.0
        xmax=Xvox[k]+s/2.0
        ymin=Yvox[k]-s/2.0
        ymax=Yvox[k]+s/2.0
        zmin=Zvox[k]-s/2.0
        zmax=Zvox[k]+s/2.0
        bound_x = np.logical_and(XYZ[:,0]>=xmin, XYZ[:,0]<=xmax)
        bound_y = np.logical_and(XYZ[:,1]>=ymin, XYZ[:,1]<=ymax)
        bound_z = np.logical_and(XYZ[:,2]>=zmin, XYZ[:,2]<=zmax)
        bb_filter = np.logical_and(bound_x, bound_y)  
        bb_filter = np.logical_and(bb_filter, bound_z)
        pos = np.array(np.where(bb_filter))
        if pos.size >=3:
            i = i + 1
            df = data.iloc[pos[0,:]]
            Dec= get_geom_feat(df)
            #Se agregan las columnas con sus respectivos descriptores en cada region
            data.loc[pos[0,:],'D1'] = Dec[0]
            data.loc[pos[0,:],'D2'] = Dec[1]
            data.loc[pos[0,:],'D3'] = Dec[2]
            data.loc[pos[0,:],'D4'] = Dec[3]
            data.loc[pos[0,:],'D5'] = Dec[4]
            data.loc[pos[0,:],'D6'] = Dec[5]
            data.loc[pos[0,:],'D7'] = Dec[6]
            data.loc[pos[0,:],'D8'] = Dec[7]
            data.loc[pos[0,:],'D9'] = Dec[8]
            data.loc[pos[0,:],'D10'] = Dec[9]
            data.loc[pos[0,:],'D11'] = Dec[10]
            data.loc[pos[0,:],'D12'] = Dec[11]
            data.loc[pos[0,:],'D13'] = Dec[12]
            #Descriptores aqui
            pointsN = pos.size
            POINTS_N.append(pointsN)
    print "Evaluated Voxels: " + str(samples)
    wrfile.close()
    return data    

if __name__ == "__main__":
    # main folders
    archivo1LOG = '/home/camila/Escritorio/kinect/Obstaculos/carpetalog/anden.log'
    archivo1BT  = '/home/camila/Escritorio/kinect/Obstaculos/vrml/anden.bt.wrl'
    DATA = features_calc(archivo1LOG,archivo1BT)  
    
    
    
    #Figura 2 Linealidad
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D1,linewidth=0)    
    plt.show()
    #Fifura 3 Planaridad
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D2,linewidth=0)    
    plt.show()
    #Figura 4 Esfericidad
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D3,linewidth=0)    
    plt.show()
    #Figura 5 Omnivarianza
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D4,linewidth=0)    
    plt.show()
    #Figura 6 Anisotropia
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D5,linewidth=0)    
    plt.show()
    #Figura 7 Eigenentropia
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D6,linewidth=0)    
    plt.show()
    #Figura 8 Sumatoria
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D7,linewidth=0)    
    plt.show()
    #Figura 9 Cambio de curvatura
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D8,linewidth=0)    
    plt.show()
    #Figura 10 Anguloo Phi
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D9,linewidth=0)    
    plt.show()
    #Figura 11 Angulo Theta
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D10,linewidth=0)    
    plt.show()
    #Figura 12 Color rojo
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D11,linewidth=0)    
    plt.show()
    #Figura 13 Color verde
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D12,linewidth=0)    
    plt.show()
    #Figura 14 Color Azul
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=DATA.D13,linewidth=0)    
    plt.show()
    #...
    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(DATA.X, DATA.Y, DATA.Z, s=10, c=(DATA.B-DATA.R-DATA.R),linewidth=0)    
    plt.show()
    
    