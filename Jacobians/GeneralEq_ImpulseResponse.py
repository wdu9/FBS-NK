# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:47:58 2021

@author: wdu
"""
import sympy as sym

from sympy import *
from sympy import  Sum, factorial, oo, IndexedBase, Function
from sympy.abc import i, k, m, n, x


import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
 

#----------------------------------------------

'''
#Derivatives of multivariable function
p=2
t=.5
c , n = sym.symbols('c n')
f = ((c*(1-n)**t)**(1-p) - 1)/(1-p)
 
#Differentiating partially w.r.t x
derivative_f = f.diff(c)
print(derivative_f)

f1 = lambdify([x,y], derivative_f )

p=2
t=.5
c , n = sym.symbols('c n')
h = ((c*(1-n)**t)**(1-p) - 1)/(1-p)
 
#Differentiating partially w.r.t x
derivative_fh = h.diff(n)
print(derivative_fh)



#-----------------------------------------------------------
#series 


w = [1,2,3]
w = IndexedBase('w')

h = Sum( w[n]**2 , (n, 1, 3)).doit()
'''

#------------------------------------------------------------------------------

CJAC=loadmat('AltCJAC')
CJAC=list(CJAC.items())
CJAC=np.array(CJAC)
CJAC = CJAC[3][1].T

CJACW=loadmat('AltCJACW')
CJACW=list(CJACW.items())
CJACW=np.array(CJACW)
CJACW = CJACW[3][1].T

CJACN=loadmat('AltCJACN')
CJACN=list(CJACN.items())
CJACN=np.array(CJACN)
CJACN = CJACN[3][1].T



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

tau = 0.1656344537815126
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
#lambda_W = .899 #probability a firm won't be able to change wage
#lambda_P = .926  #probability a firm won't be able to change wage

lambda_W = .75 #probability a firm won't be able to change wage
lambda_P = .75  #probability a firm won't be able to change wage

Lambda = (1-lambda_P)*(1-(lambda_P/(1+rstar)))/lambda_P
ParamW = ( (1-lambda_W) / lambda_W) * ( 1 - LivPrb * lambda_W )



#Policy
phi = 0
phi_y = 0





#------------------------------------------------------------------------------


# Shock Parameters       
Z = .01 # Initial Productivity shock
m_e = .01 # Initial Monetary Policy Shock
p=.96 # AR1 Coefficient


ZshkList=[]
mshkList=[]

for t in range(200):
    
    ZshkList.append((p**t)*Z)
    mshkList.append((p**t)*m_e)

        
Zshks= np.array(ZshkList)
mshk = np.array(mshkList)
            

            
# Specify Shock, if Shk = 0 then productivity shock, else Shk = 1 => monetary policy Shock

Shk = 0

dZ = np.zeros((400,1))
ShkLength = 200

if Shk == 0:
    
    for i in range(ShkLength):
        dZ[i][0] = Zshks[i]
        

if Shk == 1:
    
    for i in range(ShkLength):
       dZ[i + 200][0]=mshk[i]
            


#------------------------------------------------------------------------------
# this jacobian for pi_{t+1} wrt t=>0, below is jacobian for pi_{t} wrt t=>0

J_pi_w_1 = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        if i < j:
            
            J_pi_w_1[i][j] = -Lambda*(-1/w_ss) * (1/(1+rstar)**(j-i))
            
# this jacobian for pi_{t+1} wrt t=>0, below is jacobian for pi_{t} wrt t=>0

J_pi_Z_1 = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which wage change occurs. Columns denotes period of inflation 

for j in range(200):
    
    for i in range(200):
        
        if i < j :
            J_pi_Z_1[i][j] =  -(1/(1+rstar)**(j-i))*(Lambda/Z_ss)
        

# Price inflation


J_pi_w = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        J_pi_w[i][i] = -Lambda*(-1/w_ss)
        
        if i < j:
            J_pi_w[i][j] = -Lambda * (-1/w_ss) * ( 1 / (1+rstar)**(j-i))
            
            
J_pi_Z = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
            
        J_pi_Z[i][i] =   -Lambda/Z_ss
            
        if i < j :
            
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
    
    
    

#--------------------------------------------]
#----------------------------------------------------------------------
#Wage inflation

J_piw_C = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
                
    
        J_piw_C[i][i] =  (-rho/C_ss)
            
        if i<j:
            J_piw_C[i][j] =  (-rho/C_ss)*((DiscFac*LivPrb)**(j-i)) 

J_piw_C = -ParamW*J_piw_C


J_piw_w = np.zeros((200,200)) # Jacobian of wage Inflation wrt wage

for j in range(200):
    
    for i in range(200):
        
        J_piw_w[i][i] =  -ParamW *(1/w_ss)
            
        if i<j:
            J_piw_w[i][j] = -ParamW* (1/w_ss) *((DiscFac*LivPrb)**(j-i)) 
            
            
J_piw_w = J_piw_w + np.dot(J_piw_C,CJACW)     



J_piw_N = np.zeros((200,200)) # Jacobian of wage inflation wrt N  

for j in range(200):
    
    for i in range(200):
        
        J_piw_N[i][i] =  -ParamW*(-v/N_ss ) 
            
        if i<j:
            J_piw_N[i][j] =  -ParamW*(-v/N_ss) *((DiscFac*LivPrb)**(j-i))
            
J_piw_N = J_piw_N  + np.dot(J_piw_C,CJACN) 
            
         



#-----------------------------------------------------------------------------

# Composing HU jacobian ways through DAG


#Goods Market Clearing Target
h1 = np.zeros((200,600))
h1[:,0:200] = CJAC #Partials wrt r
h1[:,200:400] =  CJACW + J_G_w #partials wrt w
h1[:,400:600] = CJACN - J_Y_N + J_G_N #Partials wrt N


# Wage Residual Target
h2 = np.zeros((200,600))
h2[:,0:200] =    - np.dot(J_piw_C, CJAC) 
h2[:,200:400] = np.identity(200)*(1/w_ss) - J_piw_w + J_pi_w


for j in range(400):
    for i in range(200):
        if 200 + i == j and j>199 and j<400 :
            h2[i,j-1] = (-1/w_ss) - (-ParamW*(-rho/C_ss)*(CJACW[i][i-1] ))
            

h2_wagelag = np.zeros((200,200))
for j in range(200):
    for i in range(199):
        if  i + 1 == j :
            h2_wagelag[i+1, i ] = (1/w_ss)  + (-ParamW*(-rho/C_ss)*(CJACW[i+1][i] ))  

                        
h2[:,200:400]= np.identity(200)*(1/w_ss) - h2_wagelag - J_piw_w + J_pi_w  
    
 
h2[:,400:600] =   -J_piw_N  #should be a negative here but it breaks things, 
    


h3_r = np.zeros((200,200))
for j in range(200):
    for i in range(199):
        if  i + 1 == j :
            h3_r[i-1, i ] = 1 # because r_{t} =r_{t+1}^{a}

# Fisher Residual Target
h3 = np.zeros((200,600))
h3[:, 0:200] = h3_r
h3[:, 200:400] = (1+rstar)*J_pi_w_1 - phi*J_pi_w 
h3[:, 400:600] = - np.dot(J_i_Y,J_Y_N)

    
h1h2= np.vstack((h1,h2))

HU = np.vstack((h1h2,h3))


#-------------------------------------------------------------------------------

# Composing HZ with DAG method


h1z  = np.zeros((200,400))
h1z[:,0:200] = - J_Y_Z

h2z = np.zeros((200,400))
h2z[:,0:200] = J_pi_Z

h3z = np.zeros((200,400))
h3z[:,0:200] = (1+rstar)*J_pi_Z_1 - phi * J_pi_Z - phi_y * J_Y_Z
h3z[:,200:400] = - J_i_v

# Stack all the matrices
h1zh2z= np.vstack((h1z,h2z))
HZ = np.vstack((h1zh2z,h3z))

#----------------------------------------------------------------------------------------
# Putting it all together 

invHU = np.linalg.inv(HU)      
G =np.dot(-invHU,HZ)
dU = np.dot(G,dZ)


dr = dU[0:200]
dw = dU[200:400]
dN = dU[400:600]


#Real Rate and Wage, Labor/hours
plt.plot(dr, label = 'Real Interest Rate')
plt.plot(dw , label = 'Real Wage')
plt.plot(dN , label = 'Labor')
plt.legend()
plt.show()


#Consumption
dC =  np.dot(CJAC,dr) + np.dot(CJACW,dw) + np.dot(CJACN,dN) 
#plt.plot(dC)
#plt.title("Consumption")
#plt.show()

#Government Spending
dG = np.dot(J_G_N,dN) + np.dot(J_G_w,dw)
#plt.plot(dG)
#plt.title("Government Spending")
#plt.show()


#output
dY = np.dot(J_Y_N, dN) + np.dot(J_Y_Z,dZ[0:200])
#plt.plot(dY)
#plt.title("Output")
#plt.show()

#price inflation
dpi = np.dot(J_pi_w,dw) + np.dot(J_pi_Z,dZ[0:200])
#plt.plot(dpi)
#plt.title("Price Inflation")
#plt.show()

#wage inflation       
dpiw = np.dot(J_piw_N,dN) + np.dot(J_piw_w,dw) +  np.dot(J_piw_C,dC)
#plt.plot(dpiw)
#plt.title("Wage Inflation")
#plt.show()

#nominal Rate
di = np.dot(J_i_pi, dpi) + np.dot(J_i_Y, dY) + np.dot(J_i_v, dZ[200:400]) 
#plt.plot(di)
#plt.title("Nominal Rate")
#plt.show()

#Dividends
dD = dY - dw*dN
#plt.plot(dD)
#plt.title("Dividends")
#plt.show()

#Stock Price


#percentages


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dr )
axs[0, 0].set_title("Real Interest Rate")
axs[1, 0].plot(100*dw/w_ss)
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dpiw)
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(100*dpi)
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
#plt.savefig("GIPR1.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dY/N_ss)
axs[0, 0].set_title("Output")
axs[1, 0].plot(100*dC/C_ss)
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dG/.19)
axs[0, 1].set_title("Government Spending")
axs[1, 1].plot(100*dN/N_ss)
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPR2.jpg", dpi=500)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dD)
axs[0, 0].set_title("Dividends")
axs[1, 0].plot(100*di)
axs[1, 0].set_title("Nominal Rate")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dZ[0:200])
axs[0, 1].set_title("Z")
axs[1, 1].plot(dZ[200:400])
axs[1, 1].set_title("v")
fig.tight_layout()



'''




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dr)
axs[0, 0].set_title("Real Interest Rate")
axs[1, 0].plot(dw/w_ss)
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dpiw)
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(dpi)
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
#plt.savefig("GIPR1.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dY/N_ss)
axs[0, 0].set_title("Output")
axs[1, 0].plot(dC/C_ss)
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dG/.19)
axs[0, 1].set_title("Government Spending")
axs[1, 1].plot(dN/N_ss)
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPR2.jpg", dpi=500)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dD/(N_ss*(1-w_ss)))
axs[0, 0].set_title("Dividends")
axs[1, 0].plot(di)
axs[1, 0].set_title("Nominal Rate")
fig.tight_layout()






rangelen = 30

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dr[0:rangelen])
axs[0, 0].set_title("Real Interest Rate")
axs[1, 0].plot(dw[0:rangelen])
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dpiw[0:rangelen])
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(dpi[0:rangelen])
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
#plt.savefig("GIPR1.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dY[0:rangelen])
axs[0, 0].set_title("Output")
axs[1, 0].plot(dC[0:rangelen])
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dG[0:rangelen])
axs[0, 1].set_title("Government Spending")
axs[1, 1].plot(dN[0:rangelen])
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPR2.jpg", dpi=500)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dD[0:rangelen])
axs[0, 0].set_title("Dividends")
axs[1, 0].plot(di[0:rangelen])
axs[1, 0].set_title("Nominal Rate")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dZ[0:200])
axs[0, 1].set_title("Z")
axs[1, 1].plot(dZ[200:400])
axs[1, 1].set_title("v")
fig.tight_layout()

'''






















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
