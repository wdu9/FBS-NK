# -*- coding: utf-8 -*-
"""


@author: wdu

python 3.8.8

econ-ark 0.11.0

numpy 1.20.2

matplotlib 3.4.1
"""

import numpy as np

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
lambda_P = .85  #probability a firm won't be able to change price


Lambda = ( (1 - lambda_P) / lambda_P ) * ( 1 - ( lambda_P / (1+rstar) ) )
ParamW = ( (1 - lambda_W) / lambda_W ) * ( 1 - DiscFac * LivPrb * lambda_W )


print('ParamW: ' + str(ParamW) )
print('Lambda: ' + str(Lambda))




#Policy
phi = 0
phi_y = 0

#------------------------------------------------------------------------------

# Shock Parameters       
Z = .01 # Initial Productivity shock
m_e = .01 # Initial Monetary Policy Shock
p_m=.7 # AR1 Coefficient for monetary Policy shock
p_z =.9 # AR1 Coefficient for productivity shock


ZshkList=[]
mshkList=[]

for t in range(200):
    
    ZshkList.append((p_z**t)*Z)
    mshkList.append((p_m**t)*m_e)

        
Zshks= np.array(ZshkList)
mshk = np.array(mshkList)
            

            
# Specify Shock:
#if Shk = 0 <=> productivity shock, 
#else Shk = 1 <=> monetary policy Shock

Shk = 1

dZ = np.zeros((400,1))
ShkLength = 200

if Shk == 0:
    
    for i in range(ShkLength):
        dZ[i][0] = Zshks[i]
        

if Shk == 1:
    
    for i in range(ShkLength):
       dZ[i + 200][0]=mshk[i]
            


#------------------------------------------------------------------------------

''' 
Jacobians:
    
Jacobian matrices are square matrices whose ith row denotes the current period, 
and the jth column  denotes the period in which the the change in the variable occurs

'''

#-----------------------------------------------------------------------------

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
#------------------------------------------------------------------------------
# Dividends



J_D_N =  np.zeros((200,200))

for i in range(200):
    J_D_N[i][i]= -w_ss/Z_ss
    
    
J_D_w = np.zeros((200,200))

for i in range(200):
    J_D_w[i][i]= - N_ss/Z_ss
    
    
J_D_Y = np.zeros((200,200))

for i in range(200):
    J_D_Y[i][i] = 1
        
        
        
#------------------------------------------------------------------------------

#Stock Price

'''
qs_forward = np.zeros((200,200)) 
D_forward = np.zeros((200,200))
for i in range(199):
        qs_forward [i, i+1 ] = 1 
        D_forward [i, i+1 ] = 1 


# H( (qs_0, qs_1 , ..., qs_200), (D_0, D_1, ...,D_200 , r_0 , ...,r_200 ) )

H_U_qs = np.zeros((200,200))
H_U_qs = np.identity(200) - qs_forward/(1+rstar)


H_Z_qs = np.zeros((200,400))
H_Z_qs[:,0:200] = -D_forward/(1+rstar)
H_Z_qs[:,200:400] =  np.identity(200)*(q_ss + D_ss)/(1+rstar)**2

invHUqs = np.linalg.inv(H_U_qs) #dH_{u}^{-1}    
J_qs =np.dot(-invHUqs,H_Z_qs) # 

J_qs_D = J_qs[:,0:200] 
J_qs_r = J_qs[:,200:400]  

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
B_forward = np.zeros((200,200)) 
forward = np.zeros((200,200)) 
for i in range(199):
        B_forward[i, i+1 ] = 1 /(1+rstar)
        forward[i, i+1 ] = 1

# H( (B_0, B_1 , ..., B_200), (w_0, w_1, ...,w_200 , N_0 , ...,N_200, q^b_0 ,... , q^b_200 ) )

H_U_B = np.zeros((200,200))
H_U_B[:,0:200] = np.identity(200) - B_forward


H_Z_B = np.zeros((200,600))
H_Z_B[:,0:200] = -tau*N_ss*forward
H_Z_B[:,200:400] = -tau*w_ss*forward
H_Z_B[:,400:600] = -forward

invHUB = np.linalg.inv(H_U_B) #dH_{u}^{-1}    
J_B =np.dot(-invHUB,H_Z_B) # this is the jacobian of inflation with regards to the markup


J_B_w = J_B[:,0:200]
J_B_N = J_B[:,200:400]
J_B_qb = J_B[:,400:600]

J_qb_r = np.zeros((200,200))

for i in range(200):
    J_qb_r[i][i] = - (1/(1+rstar)**2)
    
            
'''
    
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
    
    

#------------------------------------------------------------------------------



#Price Inflation



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


#Markup

#Jacobian of markup wrt to wage
J_mup_w = np.zeros((200,200))

for i in range(200):
    
    J_mup_w[i][i] = -1/w_ss
    
#Jacobian of markup wrt to productivity
J_mup_Z = np.zeros((200,200))

for i in range(200):
    
    J_mup_Z[i][i] = 1/Z_ss
    

#Jacobian of inflation tomorrow with respect to wage today
J_pi1_pi0 = np.zeros((200,200))
for i in range(200):
    J_pi1_pi0[i][i]= (1+rstar)
    
J_pi1_mup0 = np.zeros((200,200))
for i in range(200):
    J_pi1_mup0[i][i] =(1+rstar)*Lambda
    
J_pi1_w = np.dot(J_pi1_mup0,J_mup_w) + np.dot(J_pi1_pi0,  np.dot(J_pi_mup,J_mup_w)) #derivative of inflation in t+1 wrt to wage in t
J_pi1_Z = np.dot(J_pi1_mup0,J_mup_Z) + np.dot(J_pi1_pi0,  np.dot(J_pi_mup,J_mup_Z))


#------------------------------------------------------------------------------------

##########################################################################################


#Wage Inflation


piw_forward = np.zeros((200,200)) 
for i in range(199):
        piw_forward[i, i+1 ] = LivPrb*DiscFac # because r_{t} = r_{t+1}^{a}


# H( (piw_0, piw_1 , ...,piw_200), (muw_0, muw_1, ...,muw_200) )

H_U_piw = np.zeros((200,200))
H_U_piw = np.identity(200) - piw_forward

H_Z_piw = np.zeros((200,200))
H_Z_piw = np.identity(200)*ParamW

invHUpiw = np.linalg.inv(H_U_piw) #dH_{u}^{-1}    
J_piw_muw = np.dot(-invHUpiw,H_Z_piw) # this is the jacobian of inflation with regards to the markup

J_muw_N = np.zeros((200,200))
J_muw_r = np.zeros((200,200))

#Jacobian of markup wrt to wage
J_muw_w = np.zeros((200,200))
for i in range(200):
    J_muw_w[i][i] = 1/w_ss
    
#Jacobian of markup wrt to mrs
J_muw_mrs = np.zeros((200,200))
for i in range(200):
    J_muw_mrs[i][i] = -1
    
#Jacobian of markup wrt to N
J_mrs_N = np.zeros((200,200))
for i in range(200):
    J_mrs_N[i][i] = v/N_ss
    
J_mrs_MU =  np.zeros((200,200))
for i in range(200):
    J_mrs_MU[i][i] = -1/MU


J_muw_w = J_muw_w + np.dot(J_muw_mrs,np.dot(J_mrs_MU,MUJACW))
J_muw_N = J_muw_N + np.dot(J_muw_mrs,np.dot(J_mrs_MU,MUJACN)) + np.dot(J_muw_mrs,J_mrs_N)
J_muw_r = J_muw_r + np.dot(J_muw_mrs,np.dot(J_mrs_MU,MUJAC))

#Wage inflation in t+1 wrt to t

J_piw1_piw0 = np.zeros((200,200))
for i in range(200):
    J_piw1_piw0[i][i] = 1/(DiscFac*LivPrb)
    
J_piw1_muw0 = np.zeros((200,200))
for i in range(200):
    J_piw1_muw0[i][i] = ParamW / (DiscFac*LivPrb)
    

    

#--------------------------------------------------------------------------------------


# Finance

# because r_{t} =r_{t+1}^{a}

J_ra_r = np.zeros((200,200)) # jacobian of return to mutual fund assets wrt to real interest rate
for i in range(199):
        J_ra_r[i+1, i ] = 1 # because r_{t} = r_{t+1}^{a}        



#-----------------------------------------------------------------------------

# derivative of log(w_{t-1}) wrt w_{t-1} given today is period t

h2_wagelag = np.zeros((200,200))
for i in range(199):
        h2_wagelag[i+1, i] = (1/w_ss) 



#-------------------------------------------------------------------------------------
            
#----------------------------------------------------------------

forward = np.zeros((200,200)) 
for i in range(199):
        forward [i, i+1 ] = 1 


#------------------------------------------------------------------------------
# Composing HU jacobian through DAG

#Goods Market Clearing Target
h1 = np.zeros((200,600))
h1[:,0:200] = np.dot(CJAC,J_ra_r)  #Partials wrt r
h1[:,200:400] =  CJACW   #partials wrt w
h1[:,400:600] = CJACN  - J_Y_N  #Partials wrt N


# Wage Residual Target
h2 = np.zeros((200,600))
h2[:,0:200] =  - np.dot(J_piw_muw, J_muw_r) 
h2[:,200:400] = np.identity(200)*(1/w_ss) - h2_wagelag - np.dot(J_piw_muw,J_muw_w) + np.dot(J_pi_mup,J_mup_w)
h2[:,400:600] =   - np.dot(J_piw_muw,J_muw_N)

'''
# Wage Residual Target
h2 = np.zeros((200,600))
h2[:,0:200] =  - np.dot(J_piw1_muw0, J_muw_r) 
h2[:,200:400] = forward*(1/w_ss) - np.identity(200)*(1/w_ss) - np.dot(J_piw1_muw0,J_muw_w) + np.dot(J_pi1_mup0,J_mup_w)
h2[:,400:600] =   - np.dot(J_piw1_muw0,J_muw_N)
'''


# Fisher Residual Target
h3 = np.zeros((200,600))
h3[:, 0:200] =  np.identity(200)
h3[:, 200:400] = (1 + rstar) * J_pi1_w - phi * np.dot(J_pi_mup,J_mup_w) 
h3[:, 400:600] = - np.dot( J_i_Y, J_Y_N )


h1h2= np.vstack((h1,h2))
HU = np.vstack((h1h2,h3))


#-------------------------------------------------------------------------------
# Composing HZ with DAG method

h1z  = np.zeros((200,400))
h1z[:,0:200] = - J_Y_Z

h2z = np.zeros((200,400))
h2z[:,0:200] = np.dot(J_pi1_mup0,J_mup_Z)

h3z = np.zeros((200,400))
h3z[:,0:200] = (1+rstar)*J_pi1_Z - phi * np.dot(J_pi_mup,J_mup_Z) - phi_y * J_Y_Z
h3z[:,200:400] = - J_i_v


# Stack all the matrices
h1zh2z= np.vstack((h1z,h2z))
HZ = np.vstack((h1zh2z,h3z))


#----------------------------------------------------------------------------------------
# Putting it all together 

invHU = np.linalg.inv(HU) #dH_{u}^{-1}    
G =np.dot(-invHU,HZ)
dU = np.dot(G,dZ)

#Endogenous Variables
dr = dU[0:200]
dw = dU[200:400]
dN = dU[400:600]



#Real Rate and Wage, Labor/hours
plt.plot(dr, label = 'Real Interest Rate')
plt.plot(dw , label = 'Real Wage')
plt.plot(dN , label = 'Labor')
plt.legend()
plt.show()


#-------------------------------------------------------------------------------

'''

B_backward = np.zeros((200,200)) 
for i in range(199):
        B_backward[i+1, i ] = 1 


# H( (B_0, B_1 , ..., B_200), (w_0, w_1, ...,w_200 , N_0 , ...,N_200, q^b_0 ,... , q^b_200 ) )

H_U_B = np.zeros((200,200))
H_U_B = B_backward - np.identity(200) /(1+rstar)

H_Z_B = np.zeros((200,600))
H_Z_B[:,0:200] = -tau*N_ss*np.identity(200)
H_Z_B[:,200:400] = -tau*w_ss*np.identity(200)
H_Z_B[:,400:600] = -np.identity(200)



invHUB = np.linalg.inv(H_U_B) #dH_{u}^{-1}    
J_B =np.dot(-invHUB,H_Z_B) # this is the jacobian of inflation with regards to the markup

J_B_w = J_B[:,0:200].T
J_B_N = J_B[:,200:400].T
J_B_qb = J_B[:,400:600].T

J_qb_r = np.zeros((200,200))
for i in range(200):
    
    J_qb_r[i][i] = - (1/(1+rstar)**2)
    
    
J_B_r = np.dot(J_B_qb,J_qb_r)

'''

#--------------------------------------------------------------------------------------
#Other endogenous Variables

# real interest rate
#dr = np.delete(dr_a , 0,0)
#dr = np.concatenate( (dr,np.array([[7.30643926e-06]]) ))

#Mutual Fund Interest Rate
dra = np.dot(J_ra_r,dr)



#Assets
dA = np.dot(AJAC,dra) + np.dot(AJACW,dw) + np.dot(AJACN,dN) 
#plt.plot(dA)

#Consumption
dC =  np.dot(CJAC, dra) + np.dot(CJACW,dw) + np.dot(CJACN,dN) 


#Marginal Utility
dMU = np.dot(MUJAC, dra) + np.dot(MUJACW,dw) + np.dot(MUJACN,dN) 

#output
dY = np.dot(J_Y_N, dN) + np.dot(J_Y_Z,dZ[0:200])


#markup
dmup = np.dot(J_mup_w,dw) + np.dot(J_mup_Z,dZ[0:200])


#price inflation
dpi = np.dot(J_pi_mup,dmup) 

#wagemarkup 
dmuw = np.dot(J_muw_N,dN) +np.dot(J_muw_w,dw) + np.dot(J_muw_r,dr)

#wage Inflation
dpiw = np.dot(J_piw_muw,dmuw)

#nominal Rate
di = np.dot(J_i_pi, dpi) + np.dot(J_i_Y, dY) + np.dot(J_i_v, dZ[200:400]) 


#Dividends
dD = np.dot(J_D_Y,dY) + np.dot(J_D_w,dw)  + np.dot(J_D_N,dN) 

#Stock Price
dq = np.dot(J_qs_D,dD) + np.dot (J_qs_r,dr)

#Bond Price
dqb  = np.dot(J_qb_r ,dr)
    
#Bonds 
dB = np.dot(J_B_w,dw) + np.dot(J_B_N,dN) + np.dot(J_B_qb,dqb)

#dB = np.dot(J_B_r,dr) + np.dot(J_B_w,dw) + np.dot(J_B_N,dN)


#Assets from asset clearing equation
dA_o = dq  + (dB/(1+rstar) - np.dot(dr, B_ss))/(1+rstar)**2

'''
plt.plot(dA, label ='assets from AJAC')
plt.plot(dA_o , label = 'assets from clearing condition')
plt.legend()
plt.show()
'''

#--------------------------------------------------------------------------------
#Impulse Response Figures

#percentages

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dr,'darkgreen' )
axs[0, 0].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[0, 0].set_title("Real Interest Rate")
axs[1, 0].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[1, 0].plot(100*dw/w_ss,'forestgreen' )
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dpiw, 'forestgreen')
axs[0, 1].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(100*dpi, 'forestgreen')
axs[1, 1].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
#plt.savefig("GIPRZ1_flexwage.jpg", dpi=500)



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dY/N_ss)
axs[0, 0].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[0, 0].set_title("Output")
axs[1, 0].plot(100*dC/C_ss)
axs[1, 0].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dB/.5)
axs[0, 1].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[0, 1].set_title("Government Bonds")
axs[1, 1].plot(100*dN/N_ss)
axs[1, 1].plot(np.zeros(200), color ='k', linestyle='dashed')
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPRZ2_flexwage.jpg", dpi=500)


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dA/A_ss, label= 'AJAC')
axs[0, 0].plot(100*dA_o/A_ss, label =' A_clear')
axs[0, 0].legend()
axs[0, 0].set_title("Assets")
axs[1, 0].plot(100*dmup/(1/w_ss))
axs[1, 0].set_title("Price Markup")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dq/q_ss)
axs[0, 1].set_title("Stock Price")
axs[1, 1].plot(100*dmuw/1.05)
axs[1, 1].set_title("wage Markup")
fig.tight_layout()
#plt.savefig("GIPRZ3_flexWage.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dD/D_ss)
axs[0, 0].set_title("Dividends")
axs[1, 0].plot(100*di)
axs[1, 0].set_title("Nominal Rate")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(dZ[0:200])
axs[0, 1].set_title("Z")
axs[1, 1].plot(dZ[200:400])
axs[1, 1].set_title("epsilon_m")
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








rangelen = 40

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
#plt.savefig("GIPRM1.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dY[0:rangelen]/N_ss, 'darkgreen')
axs[0, 0].set_title("Output")
axs[0,0].set(ylabel = '% of s.s.')
axs[1, 0].plot(100*dC[0:rangelen]/C_ss, 'darkgreen')
axs[1,0].set(ylabel = '% of s.s.')
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dA[0:rangelen]/A_ss, color = 'darkgreen')
#axs[0, 1].plot(100*dA_o[0:rangelen]/A_ss, label =' A_clear', color = 'r')
axs[0, 1].legend()
axs[0, 1].set_title("Assets")
axs[1, 1].plot(100*dN[0:rangelen]/N_ss, 'darkgreen')
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPRZ2.jpg", dpi=500)

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
axs[1, 1].plot(dZ[0:rangelen], 'darkgreen')
axs[1, 1].set_title("Z")
fig.tight_layout()
#plt.savefig("GIPRZ3.jpg", dpi=500)



rangelen = 100

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dr[0:rangelen],'darkgreen')
axs[0, 0].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[0, 0].set_title("Real Interest Rate")
axs[0,0].set(ylabel = '%')
axs[1, 0].plot(100*dw[0:rangelen]/w_ss, 'darkgreen')
axs[1, 0].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[1,0].set(ylabel = '% of s.s.')
axs[1, 0].set_title("Real Wage")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dpiw[0:rangelen], 'darkgreen')
axs[0, 1].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[0,1].set(ylabel = '%')
axs[0, 1].set_title("Nominal Wage Inflation")
axs[1, 1].plot(100*dpi[0:rangelen], 'darkgreen')
axs[1, 1].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[1,1].set(ylabel = '%')
axs[1, 1].set_title("Price Inflation")
fig.tight_layout()
#plt.savefig("GIPRM1.jpg", dpi=500)




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(100*dY[0:rangelen]/N_ss, 'darkgreen')
axs[0, 0].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[0, 0].set_title("Output")
axs[0,0].set(ylabel = '% of s.s.')
axs[1, 0].plot(100*dC[0:rangelen]/C_ss, 'darkgreen')
axs[1, 0].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[1,0].set(ylabel = '% of s.s.')
axs[1, 0].set_title("Consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(100*dA[0:rangelen]/A_ss, label= 'AJAC', color = 'darkgreen')
axs[0, 1].plot(100*dA_o[0:rangelen]/A_ss, label =' A_clear', color = 'r')
axs[0, 1].legend()
axs[0, 1].set_title("Assets")
axs[1, 1].plot(100*dN[0:rangelen]/N_ss, 'darkgreen')
axs[1, 1].plot(np.zeros(rangelen), color ='gray', linestyle='dashed')
axs[1, 1].set_title("Labor")
fig.tight_layout()
#plt.savefig("GIPRZ2.jpg", dpi=500)

'''














def wage_inflation(piw0,piw1,muw):
    
    resid = piw0 - DiscFac*LivPrb*piw1 + ParamW*(muw + 0)

    return resid

args=(1.0,1.0,1.0)
w = jac(wage_inflation,args,T=200)

print(w[0] - J_piw_muw)












#-----------------
