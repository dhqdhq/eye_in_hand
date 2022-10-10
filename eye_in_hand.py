#!/usr/bin/env python
# coding: utf-8
import transforms3d as tfs
import numpy as np
import math

def get_matrix_eular_radu(x,y,z,rx,ry,rz):
    rmat = tfs.euler.euler2mat(math.radians(rx),math.radians(ry),math.radians(rz))
    rmat = tfs.affines.compose(np.squeeze(np.asarray((x,y,z))), rmat, [1, 1, 1])
    return rmat

def skew(v):
    return np.array([[0,-v[2],v[1]],
                     [v[2],0,-v[0]],
                     [-v[1],v[0],0]])

def rot2quat_minimal(m):
    quat =  tfs.quaternions.mat2quat(m[0:3,0:3])
    return quat[1:]

def quatMinimal2rot(q):
    p = np.dot(q.T,q)
    w = np.sqrt(np.subtract(1,p[0][0]))
    return tfs.quaternions.quat2mat([w,q[0],q[1],q[2]])

hand = [  
         0.23829903669691116, 0.3710691085257341, 0.5001051244299878 ,141.38462177,51.27935673, -113.43468239,
        -0.23334318808574078 ,0.33087233978522707 ,0.4768200874197096 ,58.72787008 ,-16.36141141, -77.34010287,
          0.08608258584747974, 0.1918671450285905, 0.44678158615400093 ,102.73431051,  -4.82078209, -45.85383542,
          0.048988837208254046,0.3074269451073706,0.7494888866607448 ,89.41970842 , -5.89209507, -66.55813366,
         
        ]
camera = [
         -0.019632477691896785,0.055250535893561256,0.46983963244564325,147.42699784, -17.46478468 ,-48.20017871,
            -0.03445553872934396 ,-0.010491172354941704 ,0.44655357964904957,-152.1154274 ,  -21.8378281  ,  13.49263636,
            -0.06186156319386321, 0.02433767241880746, 0.44381620458742516 ,155.65429496, -39.59500919,  25.80985734,
            -0.11860141470293292,-0.05855610889607889,0.6523942434627338,178.4066833,-25.73181833 ,11.77196099,
         ]
# robotic -0.38948593223234257,0.02091548112797033,0.39366564790933295,-13.79477075,77.43371704,174.35136755,
#-0.38948593223234257 ,0.02091548112797033, 0.39366564790933295,34.27291607  ,-75.18551129 ,-147.8627552, 

# -0.40152364540146634,0.184930935843089,0.4203734584390752,49.78286095,67.02177523,-124.32527605,
# -0.40152364540146634,0.184930935843089 0.4203734584390752,13.4624475 ,-74.97826296,-148.13813891
def e_h_H(hand,camera):
    Hgs,Hcs = [],[]
    for i in range(0,len(hand),6):
        Hgs.append(get_matrix_eular_radu(hand[i],hand[i+1],hand[i+2],hand[i+3],hand[i+4],hand[i+5]))    
        Hcs.append(get_matrix_eular_radu(camera[i],camera[i+1],camera[i+2],camera[i+3],camera[i+4],camera[i+5]))

    Hgijs = []
    Hcijs = []
    A = []
    B = []
    size = 0
    for i in range(len(Hgs)):
        for j in range(i+1,len(Hgs)):
            size += 1
            Hgij = np.dot(np.linalg.inv(Hgs[j]),Hgs[i])
            Hgijs.append(Hgij)
            Pgij = np.dot(2,rot2quat_minimal(Hgij))
            
            Hcij = np.dot(Hcs[j],np.linalg.inv(Hcs[i]))
            Hcijs.append(Hcij)
            Pcij = np.dot(2,rot2quat_minimal(Hcij))
            
            A.append(skew(np.add(Pgij,Pcij)))
            B.append(np.subtract(Pcij,Pgij))
    MA = np.asarray(A).reshape(size*3,3)
    MB = np.asarray(B).reshape(size*3,1)
    Pcg_  =  np.dot(np.linalg.pinv(MA),MB)
    pcg_norm = np.dot(np.conjugate(Pcg_).T,Pcg_)
    Pcg = np.sqrt(np.add(1,np.dot(Pcg_.T,Pcg_)))
    Pcg = np.dot(np.dot(2,Pcg_),np.linalg.inv(Pcg))
    Rcg = quatMinimal2rot(np.divide(Pcg,2)).reshape(3,3)


    A = []
    B = []
    id = 0
    for i in range(len(Hgs)):
        for j in range(i+1,len(Hgs)):
            Hgij = Hgijs[id]
            Hcij = Hcijs[id]
            A.append(np.subtract(Hgij[0:3,0:3],np.eye(3,3)))
            B.append(np.subtract(np.dot(Rcg,Hcij[0:3,3:4]),Hgij[0:3,3:4]))
            id += 1

    MA = np.asarray(A).reshape(size*3,3)
    MB = np.asarray(B).reshape(size*3,1)
    Tcg = np.dot(np.linalg.pinv(MA),MB).reshape(3,)
    print(tfs.affines.compose(Tcg,np.squeeze(Rcg),[1,1,1]))
    return tfs.affines.compose(Tcg,np.squeeze(Rcg),[1,1,1])

