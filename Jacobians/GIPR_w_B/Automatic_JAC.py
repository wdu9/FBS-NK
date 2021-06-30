# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:17:06 2021

@author: wdu
"""
import numpy as np

import autograd

from autograd import grad



def jac(func,a,T,plusone=[False],minusone = [False]):
    
    "Second argument must always be +1 of first argument"
    
    "Function must be written variables at period t and t+1, cannot be t-1, or t+2"
   
    "plusone must be same length as (len([*args])-2) "
        
    
    
    forward = np.zeros((T,T)) 
    for i in range(T-1):
        forward [i, i+1 ] = 1
        
    backward = np.zeros((T,T))
    for i in range(T-1):
        backward [i+1, i ] = 1
        
    HU =  np.zeros((T,T))
    pd0 = grad(func)
    pd1 = grad(func,1)
    HU = np.identity(T)*pd0(*a) + forward*pd1(*a)
    

        
    n =  (len([*a])-2)
    
    if plusone == [False]:
        plusone = n*[False]
        
    if minusone== [False]:
        minusone = n*[False]
    

    HZ = np.zeros((T,n*T))
    for i in range(n):
        pdz = grad(func,i+2)
        
        if plusone[i] == True:
            HZ[:, i*T:(1+i)*T] = forward * pdz(*a)
            
        elif minusone[i] ==True:
             HZ[:, i*T:(1+i)*T] = backward * pdz(*a)  
        
        else: 
            HZ[:, i*T:(1+i)*T] = np.identity(T) * pdz(*a)
    
    
    invHU = np.linalg.inv(HU)
    J = np.dot(-invHU,HZ)
    
    JAC=[]
    for i in range(n):
        JAC.append(J[:,i*T:(1+i)*T])
    
    return JAC



def simpleJAC(func,a,T, plusone = [False] ,minusone = [False]):
    
    "First argument is variable you are taking the derivative of"
    
    "Function must be written variables at period t-2, t and t+1,  "
   
    "plusone must be same length as (len([*args])-2) "
        
    forward = np.zeros((T,T)) 
    for i in range(T-1):
        forward [i, i+1 ] = 1
        
    backward = np.zeros((T,T))
    for i in range(T-1):
        backward [i+1, i ] = 1
    
        
    simpleJAC = []
    
    n =  len([*a])
    
    if plusone == [False]:
        plusone = n*[False]
        
    if minusone== [False]:
        minusone = n*[False]
    
    
    
    for i in range(len([*a])):
        
        Pd = grad((func),i)
        
        if plusone[i] == True:
            
            J = forward*Pd(*a)
            simpleJAC.append(J)
            
        elif minusone[i] == True:
            J = backward*Pd(*a)
            simpleJAC.append(J)
        else: 
            J = np.identity(T)*Pd(*a)
            simpleJAC.append(J)
    
    
    return simpleJAC


def get_simpleJAC(flist):
    
    jac=[]
    for i in range(len(flist)):
        J = simpleJAC(flist[i])
        jac.append(J)
    return jac

        
def getJac(flist):
    
    jac=[]
    for i in range(len(flist)):
        J = jac(flist[i])
        jac.append(J)
    return jac
        



'''
def production(Z,N):
    Y = Z*N
    return Y


args = (1.0,1.19)
a = simpleJAC(production,a = args,T=200)
print(a)
'''