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



AJAC=loadmat('AltAJAC')
AJAC=list(AJAC.items())
AJAC=np.array(AJAC)
AJAC = AJAC[3][1]

AJACW=loadmat('AltAJACW')
AJACW=list(AJACW.items())
AJACW=np.array(AJACW)
AJACW = AJACW[3][1]

AJACN=loadmat('AltAJACN')
AJACN=list(AJACN.items())
AJACN=np.array(AJACN)
AJACN = AJACN[3][1]



#-------------------------------------------------------------------------------

tau = 0.16563445378151262
N_ss = 1.19
w_ss = 1/1.012
Z_ss = 1
LivPrb = .99375
DiscFac = .968
v=2

rstar = 1.048**.25 -1


rho = 2
C_ss = 1


#Phillips Curves parameters
lambda_W = .899
lambda_P = .926
Lambda = (1-lambda_P)*(1-(lambda_P/(1+rstar)))/lambda_P
ParamW = ((1-lambda_W)/lambda_W)*(1 - LivPrb*lambda_W)



#Policy
phi = 1.2
phi_y= .2



#------------------------------------------------------------------------------


            
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
            

            

Shk = 1


dZ = np.zeros((400,1))

       
if Shk == 0:
    
    for i in range(200):
        dZ[i][0] = Zshks[i]
        

if Shk == 1:
    
    for i in range(200):
       dZ[i + 200][0]=mshk[i]
            


#------------------------------------------------------------------------------
# this jacobian for pi_{t+1} wrt t=>0, below is jacobian for pi_{t} wrt t=>0
J_pi_w_1 = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        if i < j:
            
            J_pi_w_1[i][j] = -Lambda*(-1/w_ss) * (1/(1+rstar)**(j-i))
            
# this jacobian for pi_{t+1} wrt t=>0, below is jacobian for pi_{t} wrt t=>0
J_pi_Z_1 = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        if i < j :
            J_pi_Z_1[i][j] =  -(1/(1+rstar)**(j-i))*(Lambda/Z_ss)
        

# Price inflation


J_pi_w = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        J_pi_w[i][j] = -Lambda*(-1/w_ss)
        
        if i < j:
            J_pi_w[i][j] = -Lambda * (-1/w_ss) * (1/(1+rstar)**(j-i))
            
            
J_pi_Z = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
        
        if i == j :
            
            J_pi_Z[i][j] =   -Lambda/Z_ss
            
        elif i < j :
            
            J_pi_Z[i][j] =  -(1/(1+rstar)**(j-i))*(Lambda/Z_ss)
        





#----------------------------------------------------------------------------
# Output

J_Y_N = np.zeros((200,200)) # Jacobian of output response to change in labor supply

for i in range(200):
    
    J_Y_N[i][i] = Z_ss
   
    
J_Y_Z = np.zeros((200,200)) # Jacobian of output response to change in labor supply

for i in range(200):
    
    J_Y_Z[i][i] = N_ss
    

#-----------------------------------------------------------------------------
# Government Spending

J_G_w = np.zeros((200,200)) #Jacobian of G wrt wage
    
for i in range(200):
    
    J_G_w[i][i] = tau*N_ss
    

J_G_N = np.zeros((200,200)) # Jacobian of G wrt N
    
for i in range(200):
    
    J_G_N[i][i] = tau*w_ss

    
    
#-----------------------------------------------------------------------------
# Nominal Rate

J_i_pi = np.zeros((200,200)) # Jacobian of i wrt pi

for i in range(200):
    
    J_i_pi[i][i] = phi 
    
    
    
J_i_Y = np.zeros((200,200)) # Jacobian of i wrt Y

for i in range(200):
    
    J_i_Y[i][i] = phi_y 
    
J_i_v = np.zeros((200,200)) # Jacobian of i wrt v

for i in range(200):
    
    J_i_v[i][i] = 1
    
    
    

#----------------------------------------------------------------------------
# Wage Inflation


J_piw_C = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
                
        if i==j:
            J_piw_C[i][j] =  -ParamW *(-rho/C_ss)
            
        elif i<j:
            J_piw_C[i][j] =  -ParamW*(-rho/C_ss)*((DiscFac*LivPrb)**(j-i)) 

J_piw_w = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
        
        if i==j:
            
            J_piw_w[i][j] = - ParamW/w_ss
            
        
        
        elif i<j:
            
            J_piw_w[i][j] = - (ParamW/w_ss)*((DiscFac*LivPrb)**(j-i)) 
            
            
J_piw_w = J_piw_w + np.dot(J_piw_C,CJACW.T)        

J_piw_N = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
        
        if i==j:
            J_piw_N[i][j] = - ParamW * (-v/N_ss) 
            
        elif i<j:
            J_piw_N[i][j] =  -ParamW*(-v/N_ss)*((DiscFac*LivPrb)**(j-i)) 
            
J_piw_N = J_piw_N + np.dot(J_piw_C,CJACN.T)          


            


#-----------------------------------------------------------------------------

# Composing HU jacobian ways through DAG


#Goods Market Clearing Target
h1 = np.zeros((200,600))
h1[:,0:200] = CJAC.T #Partials wrt r
h1[:,200:400] =  CJACW.T - J_G_w #partials wrt w
h1[:,400:600] = CJACN.T - J_Y_N - J_G_N #Partials wrt N


# Wage Residual Target
h2 = np.zeros((200,600))
h2[:,0:200] = np.dot(J_piw_C, CJAC.T) 
h2[:,200:400] = np.identity(200)*(1/w_ss) - J_piw_w + J_pi_w

for j in range(400):
    for i in range(200):
        if i == j-1 and j>199 and j<400 :
            h2[i,i-1] = (-1/w_ss) - (rho/C_ss)*(CJACW[i-1][i] )
   
h2[:,400:600] = J_piw_N 
    

# Fisher Residual Target
h3 = np.zeros((200,600))
h3[:, 0:200] = np.identity(200)
h3[:, 200:400] = (1+rstar)*J_pi_w_1 - phi*J_pi_w 
h3[:, 400:600] = -np.dot(J_i_Y,J_Y_N)

    
h1h2= np.vstack((h1,h2))

HU = np.vstack((h1h2,h3))


#-------------------------------------------------------------------------------

# Composing HZ with DAG method


h1z  = np.zeros((200,400))
h1z[:,0:200] = - J_Y_Z

h2z = np.zeros((200,400))
h2z[:,0:200] = J_pi_Z

h3z = np.zeros((200,400))
h3z[:,0:200] = (1+rstar)*J_pi_Z_1 - phi*J_pi_Z - phi_y*J_Y_Z
h3z[:,200:400] = -J_i_v

h1zh2z= np.vstack((h1z,h2z))

HZ = np.vstack((h1zh2z,h3z))

#----------------------------------------------------------------------------------------

invHU = np.linalg.inv(HU)      

dU = np.dot(np.dot(-invHU,HZ),dZ)


dr = dU[0:200]
dw = dU[200:400]
dN = dU[400:600]

plt.plot(dr, label = 'Interest Rate')
plt.plot(dw , label = 'Wage')
plt.plot(dN , label = 'Labor')
plt.legend()
plt.show()


#Consumption
dC =  np.dot(CJAC,dr) + np.dot(CJACW,dw) + np.dot(CJACN,dN)
plt.plot(dC)
plt.title("Consumption")
plt.show()

#Government Spending
dG = np.dot(J_G_N, dN) + np.dot(J_G_w,dw)
plt.plot(dG)
plt.title("Government Spending")
plt.show()


#output
dY = np.dot(J_Y_N, dN) + np.dot(J_Y_Z,dZ[0:200])
plt.plot(dY)
plt.title("Output")
plt.show()

#price inflation
dpi = np.dot(J_pi_w,dw) + np.dot(J_pi_Z,dZ[0:200])
plt.plot(dpi)
plt.title("Price Inflation")
plt.show()

#wage inflation       
dpiw = np.dot(J_piw_C,dC) + np.dot(J_piw_N,dN) + np.dot(J_piw_w,dw)
plt.plot(dpiw)
plt.title("Wage Inflation")
plt.show()

#nominal Rate
di = np.dot(J_i_pi, dpi) + np.dot(J_i_Y, dY) + np.dot(J_i_v, dZ[200:400]) 
plt.plot(di)
plt.title("Nominal Rate")
plt.show()




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dr)
axs[0, 0].set_title("Interest Rate")
axs[1, 0].plot(dw)
axs[1, 0].set_title("Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dpiw)
axs[0, 1].set_title("Wage Inflation")
axs[1, 1].plot(dpi)
axs[1, 1].set_title("Inflation")
fig.tight_layout()




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dY)
axs[0, 0].set_title("Output")
axs[1, 0].plot(dC)
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dG)
axs[0, 1].set_title("Government")
axs[1, 1].plot(dN)
axs[1, 1].set_title("Labor")
fig.tight_layout()

























'''



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
            H_u[i][200 + j] =  - ParamW*(LivPrb**j)*((1/w_ss))  +  PsumMrs(int((i-1)/3), j, 1 ) - (Lambda/(1+rstar)**(j-int((i-1)/3)))*(-1/w_ss) 
            H_u[i][400 + j] = ParamW*(LivPrb**(j-int((i-1)/3)))*(-v/N_ss) + PsumMrs(int((i-1)/3), j, 2 )
           
    H_u[i][200] = 1/w_ss + ParamW*1/w_ss +  PsumMrs(int((i-1)/3),0, 1 )+ Lambda*(-1/w_ss)
    
    #--------------------------------------------------------------------------    
    
for i in H3:
    
    for j in range(200):
        
        
        if int((i-2)/3) == j:
             H_u[i][j] = 1
             H_u[i][200 + j] = phi*Lambda*(1/w_ss)
             H_u[i][400 + j] = phi_y*Z_ss
             
        
        elif int((i-2)/3) < j:
            H_u[i][j] = 0
            H_u[i][200 + j] = -(Lambda/(1+rstar)**j)*(1/w_ss)*(1+rstar) + (phi*Lambda/(1+rstar)**(j-int((i-2)/3)))*(1/w_ss) 
            H_u[i][400 + j] = 0
            
        elif int((i-2)/3) > j:
            H_u[i][j] = 0
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
        
        if int((i-1)/3) == j:
            
            H_z[i][j] = - Lambda/Z_ss
        
        if int((i-1)/3) < j:
            
            H_z[i][j] = - (Lambda / (1 + rstar)**(j-int((i-1)/3)))*(1/Z_ss)

#------------------------------------------------------------------------------
        
            
for i in H3:
    
    for j in range(200):
        
        if int((i-2)/3) == j :
            
            H_z[i][j] =   phi*Lambda/Z_ss - phi_y*N_ss
            H_z[i][200 + j] = -1
            
        if int((i-2)/3) < j :
            
            H_z[i][j] =  -(1+rstar)*(1/(1+rstar)**(j-int((i-2)/3)))*(Lambda/Z_ss) + (1/(1+rstar)**(j-int((i-2)/3)))*phi*Lambda/Z_ss - phi_y*N_ss
           
#----------------------------------------------------------------------------------           
            


            
Inv_H_u = np.linalg.inv(H_u)            

dU = np.dot( np.dot(-Inv_H_u,H_z),dZ)

dr = dU[0:200]
dw = dU[200:400]
dN = dU[400:600]

plt.plot(dr, label = 'Interest Rate')
plt.plot(dw , label = 'Wage')
plt.plot(dN , label = 'Labor')
plt.legend()
plt.show()



#Consumption
dC =  np.dot(CJAC,dr) + np.dot(CJACW,dw) + np.dot(CJAC,dN)
plt.plot(dC)
plt.title("Consumption")
plt.show()

#Government Spending
dG = np.dot(J_G_N, dN) + np.dot(J_G_w,dw)
plt.plot(dG)
plt.title("Government Spending")
plt.show()


#output
dY = np.dot(J_Y_N, dN) + np.dot(J_Y_Z,dZ[0:200])
plt.plot(dY)
plt.title("Output")
plt.show()

#price inflation
dpi = np.dot(J_pi_w,dw) + np.dot(J_pi_Z,dZ[0:200])
plt.plot(dpi)
plt.title("Price Inflation")
plt.show()

#wage inflation       
dpiw = np.dot(J_piw_C,dC) + np.dot(J_piw_N,dN) + np.dot(J_piw_w,dw)
plt.plot(dpiw)
plt.title("Wage Inflation")
plt.show()

#nominal Rate
di = np.dot(J_i_pi, dpi) + np.dot(J_i_Y, dY) + np.dot(J_i_v, dZ[200:400]) 
plt.plot(di)
plt.title("Nominal Rate")
plt.show()





fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dr)
axs[0, 0].set_title("Interest Rate")
axs[1, 0].plot(dw)
axs[1, 0].set_title("Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dpiw)
axs[0, 1].set_title("Wage Inflation")
axs[1, 1].plot(dpi)
axs[1, 1].set_title("Inflation")
fig.tight_layout()




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dY)
axs[0, 0].set_title("Output")
axs[1, 0].plot(dC)
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dG)
axs[0, 1].set_title("Government")
axs[1, 1].plot(dN)
axs[1, 1].set_title("Labor")
fig.tight_layout()





'''

#-----------------
