# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 00:22:14 2021

@author: wdu
"""

import numpy as np
import autograd.numpy as np
from autograd import grad

#Agent parameters
LivPrb = .99375
DiscFac = .977
v = 2
rho = 2

#Aggregate State Variables
mho = .05
tau = 0.1656344537815126
N_ss = 1.2255973765554078
w_ss =  1/1.012
Z_ss = 1
G_ss = .19
C_ss = 1.0299752212238078
A_ss = 1.705471862072535
rstar = 1.05**.25 - 1
MU= 1.704943983232289
u = 0.0954
B_ss = 0.5

D_ss = (N_ss - 1)*w_ss

q_ss = N_ss * ( 1 - w_ss ) / rstar
q_ss= 1.1855898108103429

lambda_W = .8 #probability a firm won't be able to change wage
lambda_P = .85 #probability a firm won't be able to change price


Lambda = ( (1 - lambda_P) / lambda_P ) * (1 - ( lambda_P / (1+rstar) ) )
ParamW = ( (1 - lambda_W) / lambda_W ) * ( 1 - DiscFac * LivPrb * lambda_W )


forward = np.zeros((200,200)) 
for i in range(199):
        forward [i, i+1 ] = 1 



# H( (qs_0, qs_1 , ..., qs_200), (D_0, D_1, ...,D_200 , r_0 , ...,r_200 ) )

H_U_qs = np.zeros((200,200))
H_U_qs = np.identity(200) - forward/(1+rstar)


H_Z_qs = np.zeros((200,400))
H_Z_qs[:,0:200] = -forward/(1+rstar)
H_Z_qs[:,200:400] =  np.identity(200)*(q_ss + D_ss)/(1+rstar)**2

invHUqs = np.linalg.inv(H_U_qs) #dH_{u}^{-1}    
J_qs =np.dot(-invHUqs,H_Z_qs) # 

J_qs_D = J_qs[:,0:200] 
J_qs_r = J_qs[:,200:400]  




pi_forward = np.zeros((200,200)) 
for i in range(199):
        pi_forward[i, i+1 ] = 1/(1+rstar) # because r_{t} = r_{t+1}^{a}

# H( (pi_0, pi_1 , ...,pi_200), (mu_0, mu_1, ...,mu_200) )

H_U_pi = np.zeros((200,200))
H_U_pi = np.identity(200) - pi_forward

H_Z_pi = np.zeros((200,200))
H_Z_pi = np.identity(200)*Lambda


invHUpi = np.linalg.inv(H_U_pi) #dH_{u}^{-1}    
J_pi_mup =np.dot(-invHUpi,H_Z_pi) # this is the jacobian of inflation with regards to the markup




def stock(qs0, qs1, D , r ):
    """Second argument must always be first argument +1"""
    
    resid = qs0 - (qs1 + D)/(1+r)

    return resid



args =(q_ss , q_ss, D_ss,rstar)

pd1 = grad(stock,1)
pd1(*args)



def jac(func,args,plusone,T):
    
    "Second argument must always be +1 of first argument"
    
    "Function must be written variables at period t and t+1, cannot be t-1, or t+2"
   
    "plusone must be same length as (len([*args])-2) "
        
    forward = np.zeros((T,T)) 
    for i in range(T-1):
        forward [i, i+1 ] = 1
        
    HU =  np.zeros((T,T))
    pd0 = grad(func)
    pd1 = grad(func,1)
    HU = np.identity(T)*pd0(*args) + forward*pd1(*args)
    

        
    n =  (len([*args])-2)
    HZ = np.zeros((T,n*T))
    for i in range(n):
        pdz = grad(func,i+2)
        
        if plusone[i] ==False:
            HZ[:, i*T:(1+i)*T] = np.identity(T) * pdz(*args)
        
        else: 
            HZ[:, i*T:(1+i)*T] = forward * pdz(*args)
    
    
    invHU = np.linalg.inv(HU)
    J = np.dot(-invHU,HZ)
    
    JAC=[]
    for i in range(n):
        JAC.append(J[:,i*T:(1+i)*T])
    
    return JAC





p =[True,False]
args =(q_ss , q_ss, D_ss,rstar)
a = jac(stock,args,plusone=p,T=200)
print(a)


def price_inflation(pi0,pi1,mup):
    
    resid = pi0 - pi1/(1+rstar) + Lambda*(mup)
    
    return resid

args = (0.0,0.0,1.0)
p = [False]

b =jac(price_inflation,args,plusone=p,T=200)
    

print(b[0]- J_pi_mup)

def production(Z,N):
    Y = Z*N
    return Y



def simpleJAC(func,args,plusone,T):
    
    "First argument is variable you are taking the derivative of"
    
    "Function must be written variables at period t and t+1, cannot be t-1, or t+2"
   
    "plusone must be same length as (len([*args])-2) "
        
    forward = np.zeros((T,T)) 
    for i in range(T-1):
        forward [i, i+1 ] = 1
        
    simpleJAC = []
    
    for i in range(len([*args])):
        
        Pd = grad((func),i)
        
        if plusone[i] == False:
            J = np.identity(T)*Pd(*args)
            simpleJAC.append(J)
        else: 
            J = forward*Pd(*args)
            simpleJAC.append(J)
    
    
    return simpleJAC

args = (1.0,1.19)    
p=[False,False]
    
c = simpleJAC(production,args,plusone=p,T=200)

print(c)









'''
def my_sum(*integers):
    result = 0
    a = len([*integers])
    for x in integers:
        result += x
    return result,a

print(my_sum(1,2,3,4))
'''