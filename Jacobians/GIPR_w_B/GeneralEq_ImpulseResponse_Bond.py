# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:04:59 2021

@author: wdu


Created on Fri May 28 16:47:58 2021

@author: wdu

python 3.8.8

econ-ark 0.11.0

numpy 1.20.2

matplotlib 3.4.1
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
 
#-------------------------------------------------------------------------------

#Agent parameters
LivPrb = .99375
DiscFac = .977
v = 2
rho = 2

#Aggregate State Variables
mho = .05
tau = 0.1656344537815126
N_ss = 1.2255973765554078
w_ss = 1/1.012
Z_ss = 1
G_ss = .19
C_ss = 1.0299752212238078
A_ss = 1.705471862072535
rstar = 1.05**.25 - 1
MU= 1.704943983232289
u = 0.0954
B_ss = 0.5

D_ss = (N_ss -1)*w_ss

q_ss = N_ss * ( 1 - w_ss ) / rstar
q_ss= 1.1855898108103429

#Phillips Curves parameters
lambda_W = .8 #probability a firm won't be able to change wage
lambda_P = .85  #probability a firm won't be able to change price

#when lambda_W = .0000000005 and lambda_P = .85, we see labor fall , consumption and output rise, and slight price inflation.
# My conjecture is that since firms are not flexible enough to lower their price, they must fire people since theyre productivity
# has risen, however because of the strong consumption response to changes in wage,interest rate and wages, since the wage falls
# output still rises strongly, why does the real wage rise? because the nominal wage  rises, why does the nominal wage rise?
# Because consumption rises inducing a rise in the MRS leading to a rise in the wage

#I think the reason why the graphs flip when we set wage stickiness to be 'stickier' than price stickiness is because if nominal wages fall more it may induce the real wage to flip direction and than inverting everything
# so for example, for a monetary policy shock,  we know for lambda_P = .85 and lambda_W = .8 will lead to plausible impulse responses for a monetary contraction
# however if we make lambda_P = .75 and lambda_W = .8 then the results flip. What is notable is that the real wage rises now and I believe this is what is driving the change
# Note for monetary policy shocks markups move in the same direction , WHY DOES IT FLIP??! Because Real wage?!!

#NOTE if I raise price stickiness , than wage stickiness can be closer to price stickiness , what I mean is with lambda_P = .75 and lambda_W = .75, its the wrong direction, 
# with lambda_P = .85 and lambda_W = .85, its the right direction, with lambda_P = .85 and lambda_W = .88 its the right direction but now the effects are very strong, (10 percent fall in real wage and consumption)
# lambda_P = .9 and lambda_W = .93 values are the right direction and consumption falls by about 3 percent. so the higher is lambda P the better


Lambda = ( (1 - lambda_P) / lambda_P ) * (1 - ( lambda_P / (1+rstar) ) )
ParamW = ( (1 - lambda_W) / lambda_W ) * ( 1 - DiscFac * LivPrb * lambda_W )

#Policy
phi = 0
phi_y = 0

#------------------------------------------------------------------------------

# Shock Parameters       
Z = .01 # Initial Productivity shock
m_e = .01 # Initial Monetary Policy Shock
p_m=.5 # AR1 Coefficient for monetary Policy shock
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

#Consumption Jacobians

CJAC=loadmat('AltCJAC_w_B')
CJAC=list(CJAC.items())
CJAC=np.array(CJAC)
CJAC = CJAC[3][1].T #Consumption Jacobian to interest rate

CJACW=loadmat('AltCJACW_w_B')
CJACW=list(CJACW.items())
CJACW=np.array(CJACW)
CJACW = CJACW[3][1].T #Consumption Jacobian to Wage

CJACN=loadmat('AltCJACN_w_B')
CJACN=list(CJACN.items())
CJACN=np.array(CJACN)
CJACN = CJACN[3][1].T #Consumption Jacobian to Labor



#Transpose as the Jacobian matrix columns should indicate the period of the variable in which the derivative is with respect to.
#e.g. for CJAC.T, the first column is the derivative with respect to a change in interest rate in period 0

#-----------------------------------------------------------------------------

#Marginal Utility Jacobians

MUJAC=loadmat('AltMUJAC_w_B')
MUJAC=list(MUJAC.items())
MUJAC = np.array(MUJAC)
MUJAC = MUJAC[3][1].T

MUJACW=loadmat('ALTMUJACW_w_B')
MUJACW=list(MUJACW.items())
MUJACW = np.array(MUJACW)
MUJACW = MUJACW[3][1].T

MUJACN=loadmat('ALTMUJACN_w_B')
MUJACN=list(MUJACN.items())
MUJACN = np.array(MUJACN)
MUJACN = MUJACN[3][1].T

#-----------------------------------------------------------------------------

# Asset Jacobians

AJAC=loadmat('AltAJAC_w_B')
AJAC=list(AJAC.items())
AJAC=np.array(AJAC)
AJAC = AJAC[3][1].T

AJACW=loadmat('AltAJACW_w_B')
AJACW=list(AJACW.items())
AJACW=np.array(AJACW)
AJACW = AJACW[3][1].T

AJACN=loadmat('AltAJACW_w_B')
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

#Stock price     
        
def stksum (t):
    
    p = 0
    
    for i in range(200-t):
        
        p +=  - N_ss*( 1 - w_ss) / ( ( 1 + rstar)**i  * ( 1 + rstar)**2 )
        
    return p 

J_q_r =  np.zeros((200,200))

        
for j in range(200):
    for i in range(200):
        
        J_q_r[i][i] = stksum(i)
        
        if i < j :
            J_q_r[i][j] = stksum(i) / (1+rstar)**(j-i)
        

J_q_D =  np.zeros((200,200))

for j in range(200):
    for i in range(200):
        
        J_q_D[i][i] = 0 # 1 /( 1 + rstar)
        
        if i < j :
            
            J_q_D[i][j] =  1 / (( 1 + rstar)**(j-i))


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

# Price inflation


# this jacobian for pi_{t+1} wrt t=>0, below is jacobian for pi_{t} wrt t=>0

J_pi_w_1 = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        if i < j:
            
            J_pi_w_1[i][j] = -Lambda * ( -1/w_ss ) * (1 / (1+rstar) ** (j-i) )
            
            
# this jacobian for pi_{t+1} wrt t=>0, below is jacobian for pi_{t} wrt t=>0

J_pi_Z_1 = np.zeros((200,200)) # jacobian of inflation response to change in wage. Rows represent period in which wage change occurs. Columns denotes period of inflation 

for j in range(200):
    
    for i in range(200):
        
        if i < j :
            J_pi_Z_1[i][j] =  (-Lambda/Z_ss)*(1/(1+rstar)**(j-i) )
        


J_pi_w = np.zeros((200,200)) # jacobian of inflation wrt to wage. Rows represent period in which there is a wage change. Columns represent period of inflation

for j in range(200):
    
    for i in range(200):
        
        J_pi_w[i][i] = -Lambda*(-1/w_ss)
        
        if i < j:
            J_pi_w[i][j] = -Lambda * (-1/w_ss) * ( 1 / (1+rstar)**(j-i) )
            
            
J_pi_Z = np.zeros((200,200)) # Jacobian of price inflation wrt to productivity

for j in range(200):
    
    for i in range(200):
            
        J_pi_Z[i][i] =   -Lambda/Z_ss
            
        if i < j :
            
            J_pi_Z[i][j] =  (-Lambda/Z_ss) * ( 1 / (1+rstar)**(j-i))
            
#------------------------------------------------------------------------------------

#Wage inflation

J_piw_MU = np.zeros((200,200)) #Jacobian of wage inflation wrt to Marginal Utility of Consumption

for j in range(200):
    
    for i in range(200):
                
    
        J_piw_MU[i][i] =  1/MU
            
        if i<j:
            J_piw_MU[i][j] =  (1/MU) * ( (DiscFac*LivPrb) **(j-i) ) 
            

J_piw_MU = -ParamW*J_piw_MU



J_piw_w = np.zeros((200,200)) # Jacobian of wage Inflation wrt wage

for j in range(200):
    
    for i in range(200):
        
        J_piw_w[i][i] =  -ParamW * (1/w_ss)
            
        if i<j:
            J_piw_w[i][j] = -ParamW * (1/w_ss) * ( (DiscFac*LivPrb)**(j-i)) 
            
            
J_piw_w = J_piw_w + np.dot(J_piw_MU,MUJACW)     




J_piw_N = np.zeros((200,200)) # Jacobian of wage inflation wrt N  

for j in range(200):
    
    for i in range(200):
        
        J_piw_N[i][i] =  -ParamW*( - v / N_ss ) 
            
        if i < j:
            
            J_piw_N[i][j] =  - ParamW * ( -v / N_ss  ) * ( (DiscFac*LivPrb)**(j-i) )
            
            
J_piw_N = J_piw_N  + np.dot(J_piw_MU,MUJACN) 


#--------------------------------------------------------------------------------------


# Finance

# because r_{t} =r_{t+1}^{a}

J_r_ra = np.zeros((200,200)) 
for i in range(199):
        J_r_ra[i-1, i ] = 1 # because r_{t} = r_{t+1}^{a}
 
J_ra_r = np.zeros((200,200)) # jacobian of return to mutual fund assets wrt to real interest rate
for i in range(199):
        J_ra_r[i+1, i ] = 1 # because r_{t} = r_{t+1}^{a}        



#-----------------------------------------------------------------------------

# derivative of log(w_{t-1}) wrt w_{t-1} given today is period t

h2_wagelag = np.zeros((200,200))
for i in range(199):
        h2_wagelag[i, i-1] = (1/w_ss) 





#-------------------------------------------------------------------------------------
            
#----------------------------------------------------------------




#------------------------------------------------------------------------------
# Composing HU jacobian through DAG

#Goods Market Clearing Target
h1 = np.zeros((200,600))
h1[:,0:200] = np.dot(CJAC,J_ra_r)  #Partials wrt r
h1[:,200:400] =  CJACW   #partials wrt w
h1[:,400:600] = CJACN  - J_Y_N  #Partials wrt N


# Wage Residual Target
h2 = np.zeros((200,600))
h2[:,0:200] =  - np.dot(J_piw_MU, np.dot(MUJAC,J_ra_r)) 
h2[:,200:400] = np.identity(200)*(1/w_ss) - h2_wagelag - J_piw_w + J_pi_w  
h2[:,400:600] =   - J_piw_N 
    

# Fisher Residual Target
h3 = np.zeros((200,600))
h3[:, 0:200] =  np.identity(200)
h3[:, 200:400] = (1 + rstar) * J_pi_w_1 - phi * J_pi_w 
h3[:, 400:600] = - np.dot( J_i_Y, J_Y_N )


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
dY = np.dot(J_Y_N, dN) + np.dot(J_Y_Z,dZ[0:200])


#price inflation
dpi = np.dot(J_pi_w,dw) + np.dot(J_pi_Z,dZ[0:200])


#wage inflation       
dpiw = np.dot(J_piw_N,dN) + np.dot(J_piw_w,dw) + np.dot(J_piw_MU,np.dot(MUJAC,dra))


#nominal Rate
di = np.dot(J_i_pi, dpi) + np.dot(J_i_Y, dY) + np.dot(J_i_v, dZ[200:400]) 


#Dividends
dD = np.dot(J_D_Y,dY) + np.dot(J_D_w,dw)  + np.dot(J_D_N,dN) 


#Stock Price
dq = np.dot(J_q_D,dD) + np.dot (J_q_r,dr)

#Bond Price

dq_b  = -( 1/(1+rstar)**2) *dr

#Price Markup 
dmu_p = (-1/w_ss)*dw + (1/Z_ss) * dZ[0:200]


# Wage Markup
dmu_w = (1/w_ss)*dw - ( (v/N_ss)*dN - (1/MU)*dMU)


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
axs[0, 0].plot(100*dA/A_ss, label= 'AJAC')
axs[0, 0].plot(100*dA_o/A_ss, label =' A_clear')
axs[0, 0].legend()
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


















#-----------------
