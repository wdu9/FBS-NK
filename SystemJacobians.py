# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:47:58 2021

@author: wdu
"""


import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
#------------------------------------------------------------------------------

CJAC=loadmat('AltCJAC')
CJAC=list(CJAC.items())
CJAC=np.array(CJAC)
CJAC = CJAC[3][1]

CJACW=loadmat('AltCJACW')
CJACW=list(CJACW.items())
CJACW=np.array(CJACW)
CJACW = CJACW[3][1]

CJACN=loadmat('AltCJACN')
CJACN=list(CJACN.items())
CJACN=np.array(CJACN)
CJACN = CJACN[3][1]
#-------------------------------------------------------------------------------

tau = 0.16563445378151262
N_ss = 1.19
w_ss = 1/1.012
Z_ss = 1
v=2

rho = 2
C_ss = 1

lambda_W = .899
LivPrb = .99375
DiscFac = .968
lambda_P = .926
rstar = 1.048**.25 -1

Lambda = (1-lambda_P)*(1-lambda_P/(rstar+1))/lambda_P

ParamW = ((1-lambda_W)/lambda_W)*(1 - LivPrb*lambda_W)

dmrs_dr = ParamW*(rho/C_ss)*CJAC[0][0]

phi =1.5
phi_y=0

def PsumMrs(t,d,x):
    # t is the current period
    # d is the period in which there is a deviation in the interest rate
    
    result = 0
        
# =============================================================================
#     if x==1:
#         
#         for k in range(t):
#             result +=(LivPrb**k)*((rho/C_ss)*(DiscFac**k)*CJAC[t][k])
#     else:
#         
# =============================================================================

    if x == 0:       
        JAC = CJAC
    elif x==1:
        JAC =CJACW
    elif x==2:
        JAC = CJACN
        
        

    for k in range(200-t): 
        result +=(LivPrb**k)*((rho/C_ss)*JAC[d][t+k]) #k here is the value in period k when the change occurs in period t
    
    #for k in range(200-t): 
        #result +=(LivPrb**k)*((rho/C_ss)*(DiscFac**k)*JAC[d][t+k])
    result = -ParamW*result
    
    return result

#------------------------------------------------------------------------------

H_u = np.zeros((600,600))

H1=[] #target 1
for i in range(200):
    H1.append(3*i)
    
H2 =[] # target 2
for i in range(200):
    H2.append(3*i + 1)
    
H3 = [] # target 3
for i in range(200):
    H3.append(3*i + 2)

#------------------------------------------------------------------------------

# the following calculates H_1 partial derivatives w.r.t. r,w,N for all t = 0,1,2,...
# i/3 is the period, t, that we are in
# j is the period in which the deviation occurs

for i in H1:
    
    for j in range(200):
        H_u[i][j] = CJAC[j][int(i/3)]
        H_u[i][200 + j] = CJACW[j][int(i/3)]
        H_u[i][400 + j] = CJACN[j][int(i/3)]
        
    H_u[i][200 + int(i/3)] =  CJACW[int(i/3)][int(i/3)] - tau*N_ss
    H_u[i][400 + int(i/3)] = CJACN[int(i/3)][int(i/3)] - Z_ss - w_ss*tau
     
#------------------------------------------------------------------------------


for i in H2:
    for j in range(200):
                                               # (i-1)/3 is the period, t, that we are in
        H_u[i][j] = PsumMrs(int((i-1)/3) , j, 0)  # j is the period in which the deviation occurs
        

        
        if int((i-1)/3) > j:
        
           H_u[i][200 + j] = PsumMrs(int((i-1)/3), j, 1 ) #wage 
           H_u[i][400 + j] =  PsumMrs(int((i-1)/3), j, 2 ) #labor
           
        elif int((i-1)/3)  ==  j + 1: 
            
            H_u[i][200 + j] = - 1/w_ss + PsumMrs(int((i-1)/3), j, 1 )
            H_u[i][400 + j] = PsumMrs(int((i-1)/3), j, 2 )
           
        elif int((i-1)/3) ==  j:
            
            H_u[i][200 + j] = 1/w_ss - ParamW*(1/w_ss) + PsumMrs(int((i-1)/3), j, 1 )  - Lambda*(-1/w_ss) 
            H_u[i][400 + j] = ParamW* (-v/N_ss) + PsumMrs(int((i-1)/3), j, 2 ) # should a discount factor be added here?
        
        elif int((i-1)/3) < j:
            H_u[i][200 + j] =  - ParamW*(LivPrb**j)*((1/w_ss))  +  PsumMrs(int((i-1)/3), j, 1 ) - (Lambda/(1+rstar)**j)*(-1/w_ss) 
            H_u[i][400 + j] = ParamW*(LivPrb**j)*(-v/N_ss) + PsumMrs(int((i-1)/3), j, 2 )
           
    H_u[i][200] = 1/w_ss + ParamW*1/w_ss +  PsumMrs(int((i-1)/3),0, 1 )+ Lambda*(-1/w_ss)
    
    #--------------------------------------------------------------------------    
    
for i in H3:
    
    for j in range(200):
        
        
        if int((i-2)/3) == j:
             H_u[i][j] = 1
             H_u[i][200 + j] = Lambda*(-1/w_ss)*(1+rstar) - phi*Lambda*(-1/w_ss)
             H_u[i][400 + j] = phi_y
             
        
        elif int((i-2)/3) < j:
            H_u[i][200 + j] = (Lambda/(1+rstar)**j)*(-1/w_ss)*(1+rstar) - (phi*Lambda/(1+rstar)**j)*(-1/w_ss) 
            H_u[i][400 + j] = 0
            
        elif int((i-2)/3) > j:
            H_u[i][200 + j] = 0
            H_u[i][400 + j] = 0
            
        
            

#------------------------------------------------------------------------------
         
H_z = np.zeros((600,400))
            
for i in H1:
    
    for j in range(200):
        
        if int(i/3)== j:
             H_z[i][j] = - N_ss
  
#------------------------------------------------------------------------------
             
for i in H2:
    
    for j in range(200):
        
        if int((i-1)/3) ==j :
            
            H_z[i][j] = - Lambda/Z_ss
        
        if int((i-1)/3) < j:
            
            H_z[i][j] = - (Lambda / (1 + rstar)**j)*(1/Z_ss)

#------------------------------------------------------------------------------
        
            
for i in H3:
    
    for j in range(200):
        
        if int((i-2)/3) == j :
            
            H_z[i][j] =  (Lambda/Z_ss)*(1+rstar) + -phi*Lambda/Z_ss + phi_y*N_ss
            H_z[i][200 + j] = 1
            
        if int((i-2)/3) < j :
            
            H_z[i][j] =  (1/(1+rstar)**j)* (Lambda/Z_ss)*(1+rstar) + -(1/(1+rstar)**j)*phi*Lambda/Z_ss + phi_y*N_ss
            
#----------------------------------------------------------------------------------           
            
            
Z = .01
p=.9
ZshkList=[]
mshkList=[]
m_e = .01


for t in range(200):
    
    ZshkList.append((p**t)*Z)
    mshkList.append((p**t)*m_e)

        
Zshks= np.array(ZshkList)
mshk = np.array(mshkList)
            


plt.plot(mshk)
plt.plot(Zshks)
            
dZ = np.zeros((400,1))
        
for i in range(200):
   dZ[i+200][0]=mshk[i]

#for i in range(200):
    #dZ[i][0] = Zshks[i]
            
    

            
Inv_H_u = np.linalg.inv(H_u)            

dU = np.dot( np.dot(Inv_H_u,H_z),dZ)
    



plt.plot(dU[0:200])
plt.plot(dU[200:400])
plt.plot(dU[400:600])




#_-----------------------------------------------------------------------------

J_pi_w = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        J_pi_w[i][i] = -Lambda*(-1/w_ss)
        
        if i < j:
            J_pi_w[i][j] = -Lambda*(-1/w_ss) * (1/(1+rstar)**j)
            
    

#-----------------------------------------------------------------------------

J_Y_N = np.zeros((200,200)) # Jacobian of output response to change in labor supply

for i in range(200):
    
    J_pi_w[i][i] = Z_ss
    
#-----------------------------------------------------------------------------
J_G_w = np.zeros((200,200)) #Jacobian of G wrt wage
    
for i in range(200):
    
    J_pi_w[i][i] = tau*N_ss
    
#-----------------------------------------------------------------------------
J_G_N = np.zeros((200,200)) # Jacobian of G wrt N
    
for i in range(200):
    
    J_pi_w[i][i] = tau*w_ss
    
#-----------------------------------------------------------------------------

J_i_r = np.zeros((200,200)) # Jacobian of i wrt r



















