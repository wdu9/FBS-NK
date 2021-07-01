# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:59:48 2021

@author: wdu

"""

import autograd.numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat

from Automatic_JAC import jac,simpleJAC
 
 
#-------------------------------------------------------------------------------

#Agent parameters
LivPrb = .99375
DiscFac = .9844
v = 2
rho = 2

#Aggregate State Variables
mho = .06
tau = 0.3
N_ss = 1.4
w_ss = 1/1.012
Z_ss = 1.0
G_ss = .4
C_ss = 1
A_ss = 1.5483470631100127
rstar = 1.05**.25 - 1
MU= 1.138
u = 0.2
B_ss = 0.2

D_ss = (N_ss -1)*w_ss

q_ss = N_ss * ( 1 - w_ss ) / rstar

#Phillips Curves parameters
lambda_W = .8 #probability a firm won't be able to change wage
lambda_P = .85 #probability a firm won't be able to change price

Lambda = ( (1 - lambda_P) / lambda_P ) * (1 - ( lambda_P / (1+rstar) ) )
ParamW = ( (1 - lambda_W) / lambda_W ) * ( 1 - DiscFac * LivPrb * lambda_W )


T=200

#Policy
phi = 0
phi_y = 0

#------------------------------------------------------------------------------

# Shock Parameters       
Z = .01 # Initial Productivity shock
m_e = .01 # Initial Monetary Policy Shock
p_m=.5 # AR1 Coefficient for monetary Policy shock
p_z =.98 # AR1 Coefficient for productivity shock


ZshkList=[]
mshkList=[]

for t in range(T):
    
    ZshkList.append((p_z**t)*Z)
    mshkList.append((p_m**t)*m_e)

        
Zshks= np.array(ZshkList)
mshk = np.array(mshkList)
            

            
# Specify Shock:
#if Shk = 0 <=> productivity shock, 
#else Shk = 1 <=> monetary policy Shock

Shk = 1

dZ = np.zeros((2*T,1))
ShkLength = T

if Shk == 0:
    
    for i in range(ShkLength):
        dZ[i][0] = Zshks[i]
        

if Shk == 1:
    
    for i in range(ShkLength):
       dZ[i + T][0]=mshk[i]
            


#------------------------------------------------------------------------------

''' 
Jacobians:
    
Jacobian matrices are square matrices whose ith row denotes the current period, 
and the jth column  denotes the period in which the the change in the variable occurs

'''

#-----------------------------------------------------------------------------
f = 1
#Consumption Jacobians

CJAC=loadmat('CJAC_LP')
CJAC=list(CJAC.items())
CJAC=np.array(CJAC)
CJAC = CJAC[3][1].T #Consumption Jacobian to interest rate

CJACW=loadmat('CJACW_LP')
CJACW=list(CJACW.items())
CJACW=np.array(CJACW)
CJACW = CJACW[3][1].T #Consumption Jacobian to Wage

CJACN=loadmat('CJACN_LP')
CJACN=list(CJACN.items())
CJACN=np.array(CJACN)
CJACN = CJACN[3][1].T #Consumption Jacobian to Labor



#Transpose as the Jacobian matrix columns should indicate the period of the variable in which the derivative is with respect to.
#e.g. for CJAC.T, the first column is the derivative with respect to a change in interest rate in period 0

#-----------------------------------------------------------------------------

#Marginal Utility Jacobians

MUJAC=loadmat('MUJAC_LP')
MUJAC=list(MUJAC.items())
MUJAC = np.array(MUJAC)
MUJAC = MUJAC[3][1].T

MUJACW=loadmat('MUJACW_LP')
MUJACW=list(MUJACW.items())
MUJACW = np.array(MUJACW)
MUJACW = MUJACW[3][1].T

MUJACN=loadmat('MUJACN_LP')
MUJACN=list(MUJACN.items())
MUJACN = np.array(MUJACN)
MUJACN = MUJACN[3][1].T

#-----------------------------------------------------------------------------

# Asset Jacobians

AJAC=loadmat('AJAC_LP')
AJAC=list(AJAC.items())
AJAC=np.array(AJAC)
AJAC = AJAC[3][1].T

AJACW=loadmat('AJACW_LP')
AJACW=list(AJACW.items())
AJACW=np.array(AJACW)
AJACW = AJACW[3][1].T

AJACN=loadmat('AJACW_LP')
AJACN=list(AJACN.items())
AJACN=np.array(AJACN)
AJACN = AJACN[3][1].T

#------------------------------------------------------------------------------

funclist = []


#------------------------------------------------------------------------------
# Dividends


def dividends(Y,w,N):
    
    D = Y - w*N
    return D

args=(N_ss, w_ss, N_ss)
J_D = simpleJAC(dividends,args,T=T)

J_D_Y = J_D[0]
J_D_w = J_D[1]
J_D_N = J_D[2]

            


#----------------------------------------------------------------------------
# Output


def production(Z,N):
    Y = Z*N
    return Y


args = (Z_ss,N_ss)
J_Y = simpleJAC(production,a = args,T=T)
J_Y_Z = J_Y[0]  
J_Y_N = J_Y[1]

#-----------------------------------------------------------------------------

         



#-----------------------------------------------------------------------------
# Nominal Rate
    
def taylor(pi,Y,v):
    i = phi*pi + phi_y*Y + v
    return i

args =(0.0,N_ss,0.0)
J_i = simpleJAC(taylor,args,T=T)

J_i_pi = J_i[0]
J_i_Y = J_i[1]
J_i_v = J_i[2]

#------------------------------------------------------------------------------

def price_inflation(pi0,pi1,w,Z):
    
    resid = pi0 - pi1 / (1+rstar) + Lambda*( np.log( 1.0 / w ) + np.log(Z) )
    
    return resid

args = (0.0,0.0,w_ss,Z_ss)

J_pi =jac(price_inflation,args,T=T)
J_pi_w = J_pi[0]
J_pi_Z = J_pi[1]
    


def price_inflation_tom(pi0,w,Z):
    
    pi1 = pi0*(1+rstar) + Lambda*(np.log(1.0/w) + np.log(Z))*(1+rstar)

    return pi1

args = (0.0,w_ss,Z_ss)
J_pi1 = simpleJAC(price_inflation_tom,args,T=T)
J_pi1_pi0 = J_pi1[0]

J_pi1_w = J_pi1[1] + np.dot(J_pi1_pi0,J_pi_w)
J_pi1_Z = J_pi1[2] + np.dot(J_pi1_pi0,J_pi_Z)



            
#------------------------------------------------------------------------------------


def wageinflation(piw0,piw1,w,N,MU):
    resid = piw0 - piw1*DiscFac*LivPrb + ParamW*(np.log(w) - ( np.log( N**v) - np.log(MU)    )     )
    
    return resid

args = (0.0,0.0,w_ss,N_ss,MU)
J_piw = jac(wageinflation,args,T=T)

J_piw_w = J_piw[0]
J_piw_N = J_piw[1]
J_piw_MU = J_piw[2]


J_piw_w = J_piw_w + np.dot(J_piw_MU,MUJACW)     
J_piw_N = J_piw_N  + np.dot(J_piw_MU,MUJACN) 


#--------------------------------------------------------------------------------------

J_ra_r = np.zeros((T,T)) # jacobian of return to mutual fund assets wrt to real interest rate
for i in range(T-1):
        J_ra_r[i+1, i ] = 1 # because r_{t} = r_{t+1}^{a}  
#-----------------------------------------------------------------------------

# derivative of log(w_{t-1}) wrt w_{t-1} given today is period t

h2_wagelag = np.zeros((T,T))
for i in range(T-1):
        h2_wagelag[i+1, i] = (1/w_ss) 






#-------------------------------------------------------------------------------------



#Stock price equation 


def stock(qs0,qs1,D,r):
    resid = qs0 - (qs1+D)/(1+r)
    return resid

args = (q_ss,q_ss,D_ss,rstar)
J_qs = jac(stock,args,T=T,plusone=[True,True,True])
J_qs_D = J_qs[0]
J_qs_r = J_qs[1]
             
#----------------------------------------------------------------


def Bond(B0,B1,qb,w,N):
    
    residual = B0 - qb*B1 -tau*w*N
    return residual

args =(B_ss,B_ss,(1/(1+rstar)), w_ss, N_ss)
J_B = jac(Bond,args,T=T,plusone =[True,True,True])

J_B_qb = J_B[0]
J_B_w = J_B[1]
J_B_N = J_B[2]




def BondPrice(r):
    qb = 1/(1+r)
    return qb

args= (rstar,)
J_qb = simpleJAC(BondPrice,args,T=T)

J_qb_r = J_qb[0]

J_B_r= np.dot(J_B_qb,J_qb_r)








#------------------------------------------------------------------------------
# Composing HU jacobian through DAG

'''
#Asset Market Clearing Target
h1 = np.zeros((T,3*T))
h1[:,0:T] = np.dot(AJAC,J_ra_r) - J_qs_r - ( (J_B_r*(1+rstar) - B_ss) / (1+rstar)**2)  #Partials wrt r
h1[:,T:2*T] =  AJACW - np.dot(J_qs_D,J_D_w) - J_B_w/(1+rstar)  #partials wrt w
h1[:,2*T:3*T] = AJACN  - np.dot(J_qs_D,J_D_N) - J_B_N/(1+rstar) #Partials wrt N
'''

# Goods market Clearing
h1= np.zeros((T,3*T))
h1[:,0:T] = np.dot(CJAC,J_ra_r)  #Partials wrt r
h1[:,T:2*T] = CJACW  #partials wrt w
h1[:,2*T:3*T] = CJACN  - J_Y_N  #Partials wrt N




# Wage Residual Target
h2 = np.zeros((T,3*T))
h2[:,0:T] =  - np.dot( J_piw_MU, np.dot(MUJAC,J_ra_r) ) 
h2[:,T:2*T] = np.identity(T)*(1/w_ss) - h2_wagelag - J_piw_w + J_pi_w  
h2[:,2*T:3*T] =   - J_piw_N 
    



# Fisher Residual Target
h3 = np.zeros((T,3*T))
h3[:, 0:T] =  np.identity(T)
h3[:, T:2*T] = (1 + rstar) * J_pi1_w - phi * J_pi_w 
h3[:, 2*T:3*T] = - np.dot( J_i_Y, J_Y_N )





h1h2= np.vstack((h1,h2))
HU = np.vstack((h1h2,h3))


#-------------------------------------------------------------------------------
# Composing HZ with DAG method

h1z  = np.zeros((T,2*T))
h1z[:,0:T] = - J_Y_Z
h2z = np.zeros((T,2*T))
h2z[:,0:T] = J_pi_Z

h3z = np.zeros((T,2*T))
h3z[:,0:T] = (1+rstar)*J_pi1_Z - phi * J_pi_Z - phi_y * J_Y_Z
h3z[:,T:2*T] = - J_i_v


# Stack all the matrices
h1zh2z= np.vstack((h1z,h2z))
HZ = np.vstack((h1zh2z,h3z))


#----------------------------------------------------------------------------------------
# Putting it all together 

invHU = np.linalg.inv(HU) #dH_{u}^{-1}    
G =np.dot(-invHU,HZ)
dU = np.dot(G,dZ)

#Endogenous Variables
dr = dU[0:T]
dw = dU[T:2*T]
dN = dU[2*T:3*T]



#Real Rate and Wage, Labor/hours
plt.plot(dr, label = 'Real Interest Rate')
plt.plot(dw , label = 'Real Wage')
plt.plot(dN , label = 'Labor')
plt.legend()
plt.show()


#-------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#Other endogenous Variables

# real interest rate
#dr = np.delete(dr_a , 0,0)
#dr = np.concatenate( (dr,np.array([[7.30643926e-06]]) ))

#Mutual Fund Interest Rate
dra = np.dot(J_ra_r,dr)

#Bonds 
dB = np.dot(J_B_r,dr) + np.dot(J_B_w,dw) + np.dot(J_B_N,dN)

#Assets
dA = np.dot(AJAC,dra) + np.dot(AJACW,dw) + np.dot(AJACN,dN) 
#plt.plot(dA)

#Consumption
dC =  np.dot(CJAC, dra) + np.dot(CJACW,dw) + np.dot(CJACN,dN) 


#Marginal Utility
dMU = np.dot(MUJAC, dra) + np.dot(MUJACW,dw) + np.dot(MUJACN,dN) 


#output
dY = np.dot(J_Y_N, dN) + np.dot(J_Y_Z,dZ[0:T])


#price inflation
dpi = np.dot(J_pi_w,dw) + np.dot(J_pi_Z,dZ[0:T])


#wage inflation       
dpiw = np.dot(J_piw_N,dN) + np.dot(J_piw_w,dw) + np.dot(J_piw_MU,np.dot(MUJAC,dra))


#nominal Rate
di = np.dot(J_i_pi, dpi) + np.dot(J_i_Y, dY) + np.dot(J_i_v, dZ[T:2*T]) 


#Dividends
dD = np.dot(J_D_Y,dY) + np.dot(J_D_w,dw)  + np.dot(J_D_N,dN) #+ np.dot(J_D_Z,dZ[0:200])


#Stock Price
dq = np.dot(J_qs_D,dD) + np.dot (J_qs_r,dr)

#Bond Price

dq_b  = -( 1/(1+rstar)**2) *dr

#Price Markup 
dmu_p = (-1/w_ss)*dw + (1/Z_ss) * dZ[0:T]


# Wage Markup
dmu_w = (1/w_ss)*dw - ( (v/N_ss)*dN - (1/MU)*dMU)


dA_o = dq  + (dB/(1+rstar) - np.dot(dr, B_ss))/(1+rstar)**2

plt.plot(dA, label ='AJAC')
plt.plot(dA_o , label = 'assets act')
plt.legend()
plt.show()
#--------------------------------------------------------------------------------
#Impulse Response Figures

#percentages

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dr,'darkgreen' )
axs[0, 0].set_title("Real Interest Rate")
axs[1, 0].plot(100*dw/w_ss,'forestgreen' )
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dpiw, 'forestgreen')
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(100*dpi, 'forestgreen')
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
#plt.savefig("GIPR1.jpg", dpi=500)



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dY/N_ss)
axs[0, 0].set_title("Output")
axs[1, 0].plot(100*dC/C_ss)
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dB/.5)
axs[0, 1].set_title("Government Bonds")
axs[1, 1].plot(100*dN/N_ss)
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPR2.jpg", dpi=500)


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dA/A_ss)
axs[0, 0].set_title("Assets")
axs[1, 0].plot(100*dmu_p/(1/w_ss))
axs[1, 0].set_title("Price Markup")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dq/q_ss)
axs[0, 1].set_title("Stock Price")
axs[1, 1].plot(100*dmu_w/1.05)
axs[1, 1].set_title("wage Markup")
fig.tight_layout()
#plt.savefig("GIPR3.jpg", dpi=500)


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dD/D_ss)
axs[0, 0].set_title("Dividends")
axs[1, 0].plot(100*di)
axs[1, 0].set_title("Nominal Rate")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dZ[0:T])
axs[0, 1].set_title("Z")
axs[1, 1].plot(dZ[T:2*T])
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


'''


'''



rangelen = 30

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dr[0:rangelen],'darkgreen')
axs[0, 0].set_title("Real Interest Rate")
axs[0,0].set(ylabel = '%')
axs[1, 0].plot(100*dw[0:rangelen]/w_ss, 'darkgreen')
axs[1,0].set(ylabel = '% of s.s.')
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dpiw[0:rangelen], 'darkgreen')
axs[0,1].set(ylabel = '%')
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(100*dpi[0:rangelen], 'darkgreen')
axs[1,1].set(ylabel = '%')
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
plt.savefig("GIPRM1.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dY[0:rangelen]/N_ss, 'darkgreen')
axs[0, 0].set_title("Output")
axs[0,0].set(ylabel = '% of s.s.')
axs[1, 0].plot(100*dC[0:rangelen]/C_ss, 'darkgreen')
axs[1,0].set(ylabel = '% of s.s.')
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dA[0:rangelen]/A_ss, 'darkgreen')
axs[0, 1].set_title("Assets")
axs[1, 1].plot(100*dN[0:rangelen]/N_ss, 'darkgreen')
axs[1, 1].set_title("Labor")
fig.tight_layout()
plt.savefig("GIPRZ2.jpg", dpi=500)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dD[0:rangelen]/D_ss, 'darkgreen')
axs[0,0].set(ylabel = '% of s.s.')
axs[0, 0].set_title("Dividends")
axs[1, 0].plot(100*di[0:rangelen], 'darkgreen')
axs[1,0].set(ylabel = '% ')
axs[1, 0].set_title("Nominal Rate")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dB[0:rangelen]/B_ss, 'darkgreen')
axs[0, 1].set_title("Government Bonds")
axs[1, 1].plot(dZ[0:200], 'darkgreen')
axs[1, 1].set_title("Z")
fig.tight_layout()
plt.savefig("GIPRZ3.jpg", dpi=500)


'''








'''

#Stock price     
        
def stksum (t):
    
    p = 0
    
    for i in range(200-t):
        
        p +=  - N_ss*( 1 - w_ss) / ( ( 1 + rstar)**i  * ( 1 + rstar)**2 )
        
    return p 

J_qs_r =  np.zeros((200,200))

        
for j in range(200):
    for i in range(200):
        
        J_qs_r[i][i] = stksum(i)
        
        if i < j :
            J_qs_r[i][j] = stksum(i) / (1+rstar)**(j-i)
        

J_qs_D =  np.zeros((200,200))

for j in range(200):
    for i in range(200):
        
        J_qs_D[i][i] = 0 # 1 /( 1 + rstar)
        
        if i < j :
            
            J_qs_D[i][j] =  1 / (( 1 + rstar)**(j-i))

  
#----------------------------------------------------------------

#------------------------------------------------------------------------------
#Government Bonds


def bondsum (t):
    
    p = 0
    
    for i in range(200-t):
        
        p += (tau*w_ss*N_ss - G_ss - u*mho ) / (1+rstar)**i
        
    return p 

        

#Bond Price

J_qb_r = np.zeros((200,200))

for i in range(200):
    
    J_qb_r[i][i] = - (1 / (1+rstar)**2)


J_B_qb = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
        
        J_B_qb[i][i] = 0
        
        if i < j:
            J_B_qb[i][j] =  bondsum(i) / (1+rstar)**(j-i-1)
            
for i in range(199):
        J_B_qb[i-1, i ] = bondsum(i)
        
J_B_r = np.dot(J_B_qb,J_qb_r)



J_B_w = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
        
        J_B_w[i][i] = 0
        
        if i < j:
            J_B_w[i][j] =  (tau*N_ss ) * ( 1/( 1+rstar)**(j-i ) )
            
            
            
J_B_N = np.zeros((200,200))

for j in range(200):
    
    for i in range(200):
        
        J_B_N[i][i] = 0 
        
        if i < j:
            J_B_N[i][j] =  (tau*w_ss ) * ( 1 / ( 1+rstar)**(j-i ) )
          
'''











#-----------------
