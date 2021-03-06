# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:51:38 2021

@author: William Du

python 3.8.8

econ-ark 0.11.0

numpy 1.20.2

matplotlib 3.4.1
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform, calc_expectation
from HARK.utilities import get_percentiles, get_lorenz_shares, calc_subpop_avg
from HARK import Market, make_one_period_oo_solver

import HARK.ConsumptionSaving.ConsIndShockModel2 as ConsIndShockModel
from HARK.ConsumptionSaving.ConsIndShockModel2 import (
    ConsIndShockSolver,
    IndShockConsumerType,
    PerfForesightConsumerType,
    ConsumerSolution,
)

from HARK.interpolation import (
    CubicInterp,
    LowerEnvelope,
    LinearInterp,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA
)
from HARK.distribution import Uniform, Distribution
from HARK import MetricObject, Market, AgentType
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from HARK.utilities import plot_funcs_der, plot_funcs


# My Own file
from JAC_Utility import DiscreteDistribution2, combine_indep_dstns2





#-----------------------------------------------------------------------------


##############################################################################
###############################################################################


class FBSNK_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                   "L",
                                                   "SSPmu",
                                                   "SSWmu",
                                                   
                                                   #"wage",
                                                   "N",
                                                   
                                                   "B",
                                                 
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "jacW",
                                                   "jacN",
                                                   "jacT",
                                                   "PermShkStd",
                                                   "Ghost",
                                                   
                                                    "PermShkCount",
                                                    "TranShkCount",
                                                    "TranShkStd",
                                                    "tax_rate",
                                                    "UnempPrb",
                                                    "IncUnemp",
                                                    "G",
                                                    "DisULabor",
                                                    "InvFrisch",
                                                    "s",
    
                                                    
                                                  ]
    
    

    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 0, **kwds)


    '''
    
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        
        
        PermShkDstn_U = Lognormal(np.log(self.mu_u) - (self.L*(self.PermShkStd[0])**2)/2 , self.L*self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when unemployed
        PermShkDstn_E = MeanOneLogNormal( self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when employed
        
        
        pmf_P = np.concatenate(((1-self.UnempPrb)*PermShkDstn_E.pmf ,self.UnempPrb*PermShkDstn_U.pmf)) 
        X_P = np.concatenate((PermShkDstn_E.X, PermShkDstn_U.X))
        PermShkDstn = [DiscreteDistribution(pmf_P, X_P)]
        self.PermShkDstn = PermShkDstn 
        
        TranShkDstn_E = MeanOneLogNormal( self.TranShkStd[0],123).approx(self.TranShkCount)#Transitory Shock Distribution faced when employed
        TranShkDstn_E.X = (TranShkDstn_E.X *(1-self.tax_rate)*self.wage*self.N)/(1-self.UnempPrb)**2 #NEED TO FIX THIS SQUARE TERM #add wage, tax rate and labor supply
        
        lng = len(TranShkDstn_E.X )
        TranShkDstn_U = DiscreteDistribution(np.ones(lng)/lng, self.IncUnemp*np.ones(lng)) #Transitory Shock Distribution faced when unemployed
        
        IncShkDstn_E = combine_indep_dstns(PermShkDstn_E, TranShkDstn_E) # Income Distribution faced when Employed
        IncShkDstn_U = combine_indep_dstns(PermShkDstn_U,TranShkDstn_U) # Income Distribution faced when Unemployed
        
        #Combine Outcomes of both distributions
        X_0 = np.concatenate((IncShkDstn_E.X[0],IncShkDstn_U.X[0]))
        X_1=np.concatenate((IncShkDstn_E.X[1],IncShkDstn_U.X[1]))
        X_I = [X_0,X_1] #discrete distribution takes in a list of arrays
        
        #Combine pmf Arrays
        pmf_I = np.concatenate(((1-self.UnempPrb)*IncShkDstn_E.pmf, self.UnempPrb*IncShkDstn_U.pmf))
        
        IncShkDstn = [DiscreteDistribution(pmf_I, X_I)]
        self.IncShkDstnN = IncShkDstn

        self.IncShkDstn = IncShkDstn
        self.add_to_time_vary('IncShkDstn')
        
        
        
                
        
        
        PermShkDstn_Uw = Lognormal(np.log(self.mu_u) - (self.L*(self.PermShkStd[0])**2)/2 , self.L*self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when unemployed
        PermShkDstn_Ew = MeanOneLogNormal( self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when employed
        
        TranShkDstn_Ew = MeanOneLogNormal( self.TranShkStd[0],123).approx(self.TranShkCount)#Transitory Shock Distribution faced when employed
        TranShkDstn_Ew.X = (TranShkDstn_Ew.X *(1-self.tax_rate)*(self.wage+self.dx)*self.N)/(1-self.UnempPrb)**2 #add wage, tax rate and labor supply
        
        lng = len(TranShkDstn_Ew.X )
        TranShkDstn_Uw = DiscreteDistribution(np.ones(lng)/lng, self.IncUnemp*np.ones(lng)) #Transitory Shock Distribution faced when unemployed
        
        IncShkDstn_Ew = combine_indep_dstns(PermShkDstn_Ew, TranShkDstn_Ew) # Income Distribution faced when Employed
        IncShkDstn_Uw = combine_indep_dstns(PermShkDstn_Uw,TranShkDstn_Uw)  # Income Distribution faced when Unemployed
        
        #Combine Outcomes of both distributions
        X_0 = np.concatenate((IncShkDstn_Ew.X[0],IncShkDstn_Uw.X[0]))
        X_1=np.concatenate((IncShkDstn_Ew.X[1],IncShkDstn_Uw.X[1]))
        X_I = [X_0,X_1] #discrete distribution takes in a list of arrays
        
        #Combine pmf Arrays
        pmf_I = np.concatenate(((1-self.UnempPrb)*IncShkDstn_Ew.pmf, self.UnempPrb*IncShkDstn_Uw.pmf))
        
        IncShkDstnW = [DiscreteDistribution(pmf_I, X_I)]
        
        self.IncShkDstnW = IncShkDstnW
        self.add_to_time_vary('IncShkDstnW')
        
    '''
        
    
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G + (1 - (1/(self.Rfree) ) ) * self.B) / (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        



        TranShkDstn     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstn.pmf  = np.insert(TranShkDstn.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)   
        TranShkDstn.X  = np.insert(TranShkDstn.X*(((1.0-self.tax_rate)*self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstn     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstn = [combine_indep_dstns2(PermShkDstn,TranShkDstn)]
        self.TranShkDstn = [TranShkDstn]
        self.PermShkDstn = [PermShkDstn]
        self.add_to_time_vary('IncShkDstn')
        
        TranShkDstnW     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnW.pmf  = np.insert(TranShkDstnW.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnW.X  = np.insert(TranShkDstnW.X*(((1.0-self.tax_rate)*self.N*(self.wage + self.dx))/(1 - self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnW     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnW = [combine_indep_dstns2(PermShkDstnW,TranShkDstnW)]
        self.TranShkDstnW = [TranShkDstnW]
        self.PermShkDstnW = [PermShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
     
        
     
        TranShkDstnN     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnN.pmf  = np.insert(TranShkDstnN.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnN.X  = np.insert(TranShkDstnN.X*(((1.0-self.tax_rate)*(self.N + self.dx)*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnN     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnN= [combine_indep_dstns2(PermShkDstnN,TranShkDstnN)]
        self.TranShkDstnN = [TranShkDstnN]
        self.PermShkDstnN = [PermShkDstnN]
        self.add_to_time_vary('IncShkDstnN')
        
        TranShkDstnt    = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnt.pmf  = np.insert(TranShkDstnt.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnt.X  = np.insert(TranShkDstnt.X*(((1.0- (self.tax_rate +self.dx)) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnt     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnt= [combine_indep_dstns2(PermShkDstnt,TranShkDstnt)]
        self.TranShkDstnt = [TranShkDstnt]
        self.PermShkDstnt = [PermShkDstnt]
        self.add_to_time_vary('IncShkDstnt')
        
        
             
        TranShkDstnT     = MeanOneLogNormal(self.TranShkStd[0] +self.dx ,123).approx(self.TranShkCount)
        TranShkDstnT.pmf  = np.insert(TranShkDstnT.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnT.X  = np.insert(TranShkDstnT.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnT     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnT= [combine_indep_dstns2(PermShkDstnT,TranShkDstnT)]
        self.TranShkDstnT = [TranShkDstnT]
        self.PermShkDstnT = [PermShkDstnT]
        self.add_to_time_vary('IncShkDstnT')
    
    
        #PermShkDstnP = Lognormal(np.log(.9) - ((self.PermShkStd[0])**2)/2 , self.PermShkStd[0] , 123).approx(self.PermShkCount)
        
        
        TranShkDstnP     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnP.pmf  = np.insert(TranShkDstnP.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnP.X  = np.insert(TranShkDstnP.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnP     = MeanOneLogNormal(self.PermShkStd[0] + self.dx ,123).approx(self.PermShkCount)
        self.IncShkDstnP= [combine_indep_dstns2(PermShkDstnP,TranShkDstnP)]
        self.TranShkDstnP = [TranShkDstnP]
        self.PermShkDstnP = [PermShkDstnP]
        self.add_to_time_vary('IncShkDstnP')
    
        
    
    
        
    
    
    
FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.05**.25,                       # Interest factor on assets
    "DiscFac": 0.97,                     # Intertemporal discount factor
    "LivPrb" : [.99375],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [.03], #[(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.06, #.08                     # Probability of unemployment while working
    "IncUnemp" :  .2, #0.0954, #0.29535573122529635,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : .3, #0.16563445378151262,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other parameters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : False,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type

    # Parameters only used in simulation
    "AgentCount" : 50000,                 # Number of agents of this type
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(1.6)-(.5**2)/2,# Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     "dx"         : .1,                  #Deviation from steady state
     "jac"        : False,
     "jacW"       : False, 
     "jacN"       : False,
     "jact"       : False,
     "jacT"       : False,
     "jacPerm"    : False,
     "Ghost"      : False, 
     
     
    #New Economy Parameters
     "SSWmu" : 1.05 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.012,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : .2,                               # Net Bond Supply
     "G" : .4,
     "DisULabor": 0.8823685356415617,
     "InvFrisch": 2 ,
     "s" : 1
     }


    
###############################################################################


G=.4
t=.3 #0.16563445378151262
Inc = .2
mho=.06
r = (1.05)**.25 - 1 
B=.2 #.65

w = (1/1.012)
N = (Inc*mho + G  + (1 - (1/(1+r)) ) *B) / (w*t) 
q = ((1-w)*N)/r

A = ( B/(1+r) ) + q

print(N)
print(q)
print(A)



'''
N = 1 + G

tnew = (Inc*mho + G)/(N*w)
print(tnew)

new = (N*w*tnew - .18) / (mho)
print(new)

print(N)
print(N-G)
print(q)

'''






###############################################################################

ss_agent = FBSNK_agent(**FBSDict)
ss_agent.cycles = 0
ss_agent.dx = 0
ss_agent.T_sim = 1200
ss_agent.aNrmInitMean = np.log(A)-(.5**2)/2
#ss_agent.track_vars = ['TranShk']

target = A


NumAgents = 500000

tolerance = .01

completed_loops=0

go = True

num_consumer_types = 5     # number of types 


center = .9844 #.9773 #98 
#0.9778688120928881 for one agent
while go:
    
    discFacDispersion = 0.005
    bottomDiscFac     = center - discFacDispersion
    topDiscFac        = center + discFacDispersion
    
    tail_N = 2
    #param_dist = Lognormal(mu=np.log(center)-0.5*spread**2,sigma=discFacDispersion,tail_N=tail_N,tail_bound=[0.0,0.9], tail_order=np.e).approx(N=num_consumer_types-tail_N)
    
    #DiscFac_dist =Lognormal(mu=np.log(center)-0.5*discFacDispersion**2,sigma=discFacDispersion).approx(N=num_consumer_types-tail_N,tail_N=2, tail_bound=[0,0.9])
    DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
    DiscFac_list  = DiscFac_dist.X
    
    consumers_ss = [] 
    
    # now create types with different disc factors
    
    list_pLvl = []
    list_aNrm = []
    list_aLvl = []
    litc = []
    litMU =[]
    for i in range(num_consumer_types):
        consumers_ss.append(deepcopy(ss_agent))
        consumers_ss[i].DiscFac    = DiscFac_list[i]
        consumers_ss[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
    
    
        # solve and simulate
        consumers_ss[i].solve()
        consumers_ss[i].initialize_sim()
        consumers_ss[i].simulate()
        
        list_pLvl.append(consumers_ss[i].state_now['pLvl'])
        list_aNrm.append(consumers_ss[i].state_now['aNrm'])
        litc.append((consumers_ss[i].state_now['mNrm'] - consumers_ss[i].state_now['aNrm'])*consumers_ss[i].state_now['pLvl'])
        list_aLvl.append(consumers_ss[i].state_now['aLvl'])
        
        
        emp = consumers_ss[i].shocks['TranShk'] != ss_agent.IncUnemp
        consumers_ss[i].shocks['TranShk'][emp] = ((1-ss_agent.UnempPrb)/(ss_agent.wage * ss_agent.N * (1 - ss_agent.tax_rate)))*consumers_ss[i].shocks['TranShk'][emp]
        litMU.append(consumers_ss[i].DiscFac*consumers_ss[i].shocks['TranShk']*consumers_ss[i].state_now['pLvl']*((consumers_ss[i].state_now['mNrm'] - consumers_ss[i].state_now['aNrm'])*consumers_ss[i].state_now['pLvl'])**(- ss_agent.CRRA))

        
        print('one consumer solved and simulated')
    
    
    
    pLvl = np.concatenate(list_pLvl)
    aNrm = np.concatenate(list_aNrm)
    c = np.array(np.concatenate(litc))
    aLvl = np.concatenate(list_aLvl)

    AggA = np.mean(np.array(aLvl))
    AggC = np.mean(c)

    
    MU = np.array(np.concatenate(litMU))
    MU = np.mean(MU)
    
    MV = ss_agent.DisULabor* (ss_agent.N/(1-ss_agent.UnempPrb))**ss_agent.InvFrisch

    MRS = MV / MU
    
    AMRS = ss_agent.SSWmu * ss_agent.wage * ( 1 - ss_agent.tax_rate )
    
    AMV = AMRS * MU
    
    disu_act = AMV/((ss_agent.N/(1-ss_agent.UnempPrb))**ss_agent.InvFrisch)
    
    dif = AggA - target
    
    if AggA - target > 0 :
        
       center = center - dif/200
        
    elif AggA - target < 0: 
        center = center- dif/200
        
    else:
        break
    
    
    
    print('MRS =' + str(MRS))
    print('what it needs to be:' + str(AMRS))
    print('Assets =' + str(AggA))
    print('consumption =' + str(AggC))
    print('center =' + str(center))
    
    distance = abs(AggA - target) 
    
    completed_loops += 1
    
    print('Completed loops:' + str(completed_loops))
    
    go = distance >= tolerance and completed_loops < 1
        
print("Done Computing Steady State")


'''
for i in range(len(DiscFac_list)):
    
    DiscFac_list[i]
'''  

###############################################################################
###############################################################################

'''
m = consumers_ss[3].DiscFac*consumers_ss[3].shocks['TranShk']*(1/((consumers_ss[3].state_now['mNrm'] - consumers_ss[3].state_now['aNrm'])*consumers_ss[3].state_now['pLvl']))**1
print(m)
print(np.mean(m))

#m=(consumers_ss[3].state_now['mNrm']*consumers_ss[3].state_now['pLvl'])

p=0
for i in range(len(m)):
    if m[i] < 1:
        p+=1
print(p)      



funcs=[]
list_mLvl = []
list_mNrm = []
list_aNrm = []
for i in range(num_consumer_types):
    list_mLvl.append(consumers_ss[i].state_now['mNrm']*consumers_ss[i].state_now['pLvl'] )
    list_mNrm.append(consumers_ss[i].state_now['mNrm'])
    list_aNrm.append(consumers_ss[i].state_now['aNrm'])
    funcs.append(consumers_ss[i].solution[0].cFunc)

mNrm = np.concatenate(list_mNrm)   
mLvl = np.concatenate(list_mLvl)
aNrm = np.concatenate(list_aNrm)



x = np.linspace(0, 5, 1000, endpoint=True)

y=[]
for i in range(num_consumer_types):
    y.append(funcs[i](x))


h = np.histogram(mNrm, bins=np.linspace(0,5,num=1000))

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Cash on Hand')
plt.xlim(0,5)
ax1.plot(x, y[0], 'k' )

ax1.plot(x, y[1], 'm' )
ax1.plot(x, y[2], 'darkorange' )
ax1.plot(x, y[3], 'forestgreen' )
ax1.plot(x, y[4], 'deepskyblue' )
#ax1.plot(x, y[5], 'r' )
#ax1.plot(x, y[6], 'darkslategrey' )


ax1.set_ylim((0,.23))
ax1.set_ylabel('Consumption', color='k')


ax2= ax1.twinx()
ax2.hist(mNrm, bins=np.linspace(0,1.4,num=1000),color = 'darkviolet')
#ax2.hist(example.state_now['mNrm'],bins=np.linspace(0,1.4,num=1000),color= 'orange' )
ax2.set_ylim((0,1600))
ax2.set_ylabel('Number of Households', color='k')
#plt.savefig("Presentation.png", dpi=150)

'''


################################################################################
################################################################################


class FBSNK_JAC(FBSNK_agent):
    
    
     def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ss_agent.N #calculate SS labor supply from Budget Constraint
        



        TranShkDstn     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstn.pmf  = np.insert(TranShkDstn.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)   
        TranShkDstn.X  = np.insert(TranShkDstn.X*(((1.0-self.tax_rate)*self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstn     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstn = [combine_indep_dstns2(PermShkDstn,TranShkDstn)]
        self.TranShkDstn = [TranShkDstn]
        self.PermShkDstn = [PermShkDstn]
        self.add_to_time_vary('IncShkDstn')
        
        TranShkDstnW     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnW.pmf  = np.insert(TranShkDstnW.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnW.X  = np.insert(TranShkDstnW.X*(((1.0-self.tax_rate)*self.N*(self.wage + self.dx))/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnW     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnW = [combine_indep_dstns2(PermShkDstnW,TranShkDstnW)]
        self.TranShkDstnW = [TranShkDstnW]
        self.PermShkDstnW = [PermShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
     
        
     
        TranShkDstnN     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnN.pmf  = np.insert(TranShkDstnN.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnN.X  = np.insert(TranShkDstnN.X*(((1.0-self.tax_rate)*(self.N + self.dx)*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnN     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnN= [combine_indep_dstns2(PermShkDstnN,TranShkDstnN)]
        self.TranShkDstnN = [TranShkDstnN]
        self.PermShkDstnN = [PermShkDstnN]
        self.add_to_time_vary('IncShkDstnN')
        
        
             
        TranShkDstnt    = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnt.pmf  = np.insert(TranShkDstnt.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnt.X  = np.insert(TranShkDstnt.X*(((1.0- (self.tax_rate +self.dx)) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnt     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnt= [combine_indep_dstns2(PermShkDstnt,TranShkDstnt)]
        self.TranShkDstnt = [TranShkDstnt]
        self.PermShkDstnt = [PermShkDstnt]
        self.add_to_time_vary('IncShkDstnt')
        
        TranShkDstnP     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnP.pmf  = np.insert(TranShkDstnP.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnP.X  = np.insert(TranShkDstnP.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnP     = MeanOneLogNormal(self.PermShkStd[0] + self.dx ,123).approx(self.PermShkCount)
        self.IncShkDstnP= [combine_indep_dstns2(PermShkDstnP,TranShkDstnP)]
        self.TranShkDstnP = [TranShkDstnP]
        self.PermShkDstnP = [PermShkDstnP]
        self.add_to_time_vary('IncShkDstnP')
        
        TranShkDstnT     = MeanOneLogNormal(self.TranShkStd[0] + self.dx,123).approx(self.TranShkCount)
        TranShkDstnT.pmf  = np.insert(TranShkDstnT.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnT.X  = np.insert(TranShkDstnT.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnT     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnT= [combine_indep_dstns2(PermShkDstnT,TranShkDstnT)]
        self.TranShkDstnT = [TranShkDstnT]
        self.PermShkDstnT = [PermShkDstnT]
        self.add_to_time_vary('IncShkDstnT')

    
  
     def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        Parameters
        ----------
        none
        Returns
        -------
        none
        """
        
        
            
        for i in range(num_consumer_types):
            if self.DiscFac == DiscFac_list[i]:
                print(self.DiscFac)
                self.solution_terminal.cFunc = deepcopy(consumers_ss[i].solution[0].cFunc)
                self.solution_terminal.vFunc = deepcopy(consumers_ss[i].solution[0].vFunc)
                self.solution_terminal.vPfunc = deepcopy(consumers_ss[i].solution[0].vPfunc)
                self.solution_terminal.vPPfunc =  deepcopy(consumers_ss[i].solution[0].vPPfunc)

    
    
     def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rfree in every entry.
        Parameters
        ----------
        None
        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
    
        
        if self.jac==True or self.Ghost == True:
            RfreeNow = self.Rfree[self.t_sim ]* np.ones(self.AgentCount)
        else:
           
            RfreeNow = ss_agent.Rfree * np.ones(self.AgentCount)
            
        return RfreeNow
    


     def transition(self):
        
        
        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvlNow = pLvlPrev*self.shocks['PermShk']  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev['PlvlAgg']*self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow/self.shocks['PermShk']
        bNrmNow = ReffNow*aNrmPrev         # Bank balances before labor income
        mNrmNow = bNrmNow + self.shocks['TranShk']  # Market resources after income
        
                
        if self.t_sim == 0:
            
            for i in range(num_consumer_types):
                if  self.DiscFac == consumers_ss[i].DiscFac:

                    mNrmNow = consumers_ss[i].state_now['mNrm']
                    pLvlNow = consumers_ss[i].state_now['pLvl']
                    

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None
    


###############################################################################


params = deepcopy(FBSDict)
params['T_cycle']= 201
params['LivPrb']= params['T_cycle']*[ss_agent.LivPrb[0]]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[ss_agent.PermShkStd[0]]
params['TranShkStd']= params['T_cycle']*[ss_agent.TranShkStd[0]]
params['Rfree'] = params['T_cycle']*[ss_agent.Rfree]

###############################################################################

ghost_agent = FBSNK_JAC(**params)
ghost_agent.pseudo_terminal = False
ghost_agent.IncShkDstn = params['T_cycle']*ghost_agent.IncShkDstn


ghost_agent.del_from_time_inv('Rfree')
ghost_agent.add_to_time_vary('Rfree')

ghost_agent.T_sim = params['T_cycle']
ghost_agent.cycles = 1
ghost_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl','TranShk']
ghost_agent.dx = 0
ghost_agent.jac = False
ghost_agent.jacW = False
ghost_agent.Ghost = True
ghost_agent.PerfMITShk = True


ghosts= [] 

for i in range(num_consumer_types):
    ghosts.append(deepcopy(ghost_agent))
    ghosts[i].DiscFac   = DiscFac_list[i]
    ghosts[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])


##############################################################################
##############################################################################
     
     
listA_g = []
listH_Ag= []
listC_g = []
listH_g = []
listH_Mg =[]
listM_g = []
listH_MUg = []
listMU_g =[]

norm = ((1-ss_agent.UnempPrb)/((ss_agent.wage) * ss_agent.N * (1 - ss_agent.tax_rate)))

for k in range(num_consumer_types):
    ghosts[k].solve()
    ghosts[k].initialize_sim()
    ghosts[k].simulate()
    
    for j in range(ghost_agent.T_sim):

        emp = ghosts[k].history['TranShk'][j] != ss_agent.IncUnemp

        ghosts[k].history['TranShk'][j][emp] = norm*ghosts[k].history['TranShk'][j][emp]



    listH_g.append([ghosts[k].history['cNrm'], ghosts[k].history['pLvl']])
    listH_Ag.append(ghosts[k].history['aLvl'])
    listH_Mg.append(ghosts[k].history['mNrm'])
    listH_MUg.append(ghosts[k].history['TranShk'])
    
    
for j in range(ghost_agent.T_sim):
    
    

    litc_g=[]
    lita_g=[]
    litm_g=[]
    litMU_g=[]
    
    for n in range(num_consumer_types):
        litc_g.append(listH_g[n][0][j,:]*listH_g[n][1][j,:])
        lita_g.append(listH_Ag[n][j,:])
        litm_g.append(listH_Mg[n][j,:]*listH_g[n][1][j,:])
        
        emp = listH_MUg[n][j,:] != ss_agent.IncUnemp
        
        litMU_g.append( listH_MUg[n][j,:][emp] * listH_g[n][1][j,:][emp] * ( listH_g[n][0][j,:][emp] * listH_g[n][1][j,:][emp] )**( -ss_agent.CRRA ) )
    
    
    MU = np.array(np.concatenate(litMU_g))
    MU = np.mean(MU)
    
    Ag=np.concatenate(lita_g)
    Ag=np.mean(np.array(Ag))
    
    Cg = np.concatenate(litc_g)
    Cg = np.mean(np.array(Cg))

    Mg = np.concatenate(litm_g)
    Mg = np.mean(np.array(Mg))

    listMU_g.append(MU)
    listM_g.append(Mg)
    listA_g.append(Ag)
    listC_g.append(Cg)
    
MU_dx0 = np.array(listMU_g)
M_dx0 = np.array(listM_g)
A_dx0 = np.array(listA_g)
C_dx0 = np.array(listC_g)


plt.plot(A_dx0, label = 'Asset Steady State')
plt.legend()
plt.show()

plt.plot(C_dx0, label = 'Consumption Steady State')
plt.legend()
plt.show()


plt.plot(MU_dx0, label = 'Marginal Utility Steady State')
plt.legend()
plt.show()

print('done with Ghosts')

##################################################################################
###############################################################################
###############################################################################




#Jacobian Path


jac_agent = FBSNK_JAC(**params)
jac_agent.pseudo_terminal = False
jac_agent.PerfMITShk = True
jac_agent.jac = False
jac_agent.jacW = True
jac_agent.jacN = False
jac_agent.jact = False
jac_agent.jacT = False
jac_agent.jacPerm = False



jac_agent.IncShkDstn = params['T_cycle']*jac_agent.IncShkDstn
jac_agent.T_sim = params['T_cycle']
jac_agent.cycles = 1
jac_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl','TranShk']

if jac_agent.jac == True:
    jac_agent.dx = .001
    
    jac_agent.del_from_time_inv('Rfree')
    jac_agent.add_to_time_vary('Rfree')
    jac_agent.IncShkDstn = params['T_cycle']*ss_agent.IncShkDstn
    
    

if jac_agent.jacW == True or jac_agent.jacN == True or jac_agent.jact ==True or jac_agent.jacT ==True or jac_agent.jacPerm == True:
    jac_agent.dx = .01 #.8
    jac_agent.Rfree = ss_agent.Rfree
    jac_agent.update_income_process()


consumers = [] 

# now create types with different disc factors

for i in range(num_consumer_types):
    consumers.append(deepcopy(jac_agent))


for i in range(num_consumer_types):
        consumers[i].DiscFac    = DiscFac_list[i]
        consumers[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])


###############################################################################
###############################################################################


testSet= [0,20,50]

CHist = []
AHist = []
MHist = []
MUHist = []
pLvlHist =[]

#for i in range(params['T_cycle']):
    
    
for i in testSet:
        
        listH_C = []
        listH_A = []
        listH_M = []
        listH_MU =[]
        listMU =[]
        listC = []
        listA = []
        listM = []
        listpLvl =[]
        
        for k in range(num_consumer_types):
            
            consumers[k].s = i 
            
            if jac_agent.jac == True:
                consumers[k].Rfree = (i)*[ss_agent.Rfree] + [ss_agent.Rfree + jac_agent.dx] + (params['T_cycle'] - i - 1)*[ss_agent.Rfree]
            
            if jac_agent.jacW == True:
                consumers[k].IncShkDstn = i*ss_agent.IncShkDstn + jac_agent.IncShkDstnW + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn
                
            if jac_agent.jacN == True:
                consumers[k].IncShkDstn = i*ss_agent.IncShkDstn + jac_agent.IncShkDstnN + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn
                
            if jac_agent.jact == True:
                consumers[k].IncShkDstn = i*ss_agent.IncShkDstn + jac_agent.IncShkDstnt + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn
                
            if jac_agent.jacT == True:
                consumers[k].IncShkDstn = i*ss_agent.IncShkDstn + jac_agent.IncShkDstnT + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn
                
            if jac_agent.jacPerm == True:
                #consumers[k].PermShkStd = (i)*[ss_agent.PermShkStd[0]] + [ss_agent.PermShkStd[0] + jac_agent.dx] + (params['T_cycle'] - i - 1)*[ss_agent.PermShkStd[0]]

                consumers[k].IncShkDstn = i*ss_agent.IncShkDstn + jac_agent.IncShkDstnP + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn



            consumers[k].solve()
            consumers[k].initialize_sim()
            consumers[k].simulate()
            
            
            for j in range(jac_agent.T_sim):
                
                if jac_agent.jacW == True:
                    
                    if j == consumers[k].s + 1 :
                        norm1 =  ((1-ss_agent.UnempPrb)/( (ss_agent.wage + jac_agent.dx) * ss_agent.N * (1 - ss_agent.tax_rate)))

                    else:
                         norm1 =norm
                         
                         
                elif jac_agent.jacN == True:
                    
                    if j == consumers[k].s + 1:
                        norm1 =  ((1-ss_agent.UnempPrb)/( (ss_agent.wage) * (ss_agent.N + jac_agent.dx) * (1 - ss_agent.tax_rate) ) )
                    else:
                      norm1 =norm
                        
                      
                elif jac_agent.jact == True:
                    
                    if j == consumers[k].s + 1  :
                        norm1 =  ( ( 1-ss_agent.UnempPrb ) / (  (ss_agent.wage)  * (ss_agent.N ) * (1 -  ( ss_agent.tax_rate + jac_agent.dx ) ) ) )

                    else:
                        norm1 = norm
                        
                else:
                    norm1 =norm
                
                
                emp = consumers[k].history['TranShk'][j] != ss_agent.IncUnemp
                consumers[k].history['TranShk'][j][emp] = norm1 * consumers[k].history['TranShk'][j][emp]

            
            listH_C.append([consumers[k].history['cNrm'],consumers[k].history['pLvl']])
            listH_M.append(consumers[k].history['mNrm'])
            listH_A.append(consumers[k].history['aLvl'])
            listH_MU.append(consumers[k].history['TranShk'])


            
        for j in range(jac_agent.T_sim):
            
            litc_jac = []
            lita_jac = []
            litm_jac =  []
            litMU_jac = []
            litpLvl_jac = []
            
            for n in range(num_consumer_types):
                litc_jac.append(listH_C[n][0][j,:]*listH_C[n][1][j,:])
                lita_jac.append(listH_A[n][j,:])
                litm_jac.append(listH_M[n][j,:]*listH_C[n][1][j,:])
                litpLvl_jac.append(listH_C[n][1][j,:])
                
                emp = listH_MU[n][j,:]  != ss_agent.IncUnemp
                
                listH_C[n][1][j,:][emp]
                litMU_jac.append(listH_MU[n][j,:][emp] * listH_C[n][1][j,:][emp] *(listH_C[n][0][j,:][emp] * listH_C[n][1][j,:][emp])**(-ss_agent.CRRA))


            pLvl = np.concatenate(litpLvl_jac)
            pLvl = np.mean(np.array(pLvl))
            listpLvl.append(pLvl)

            MU = np.concatenate(litMU_jac)
            MU = np.mean(np.array(MU))
            listMU.append(MU)
            
            c = np.concatenate(litc_jac)
            c = np.mean(np.array(c))
            listC.append(c)
            
            a = np.concatenate(lita_jac)
            a = np.mean(np.array(a))
            listA.append(a)
            
            m = np.concatenate(litm_jac)
            m = np.mean(np.array(m))
            listM.append(m)
            
        pLvlHist.append(np.array(listpLvl))
        MUHist.append(np.array(listMU))
        AHist.append(np.array(listA))
        MHist.append(np.array(listM))
        CHist.append(np.array(listC))
          # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s
        print(i)


###############################################################################
###############################################################################

if jac_agent.jacW ==True:
    
    CJACW = []
    AJACW = []
    MUJACW=[]
    for i in range(params['T_cycle']):
        CJACW.append((CHist[i]-C_dx0)/jac_agent.dx)
        AJACW.append((AHist[i]-A_dx0)/jac_agent.dx)
        MUJACW.append((MUHist[i]-MU_dx0)/jac_agent.dx)
    
    savemat('CJACW.mat', mdict={'CJACW' : CJACW})
    savemat('AJACW.mat', mdict={'AJACW' : AJACW})
    savemat('MUJACW.mat', mdict={'MUJACW' : MUJACW})

if jac_agent.jacN == True:
    

    CJACN = []
    AJACN = []
    MUJACN=[]
    for i in range(params['T_cycle']):
        CJACN.append((CHist[i]-C_dx0)/jac_agent.dx)
        AJACN.append((AHist[i]-A_dx0)/jac_agent.dx)
        MUJACN.append((MUHist[i]-MU_dx0)/jac_agent.dx)
        
    savemat('CJACN_LP.mat', mdict={'CJACN_LP' : CJACN})
    savemat('AJACN_LP.mat', mdict={'AJACN_LP' : AJACN})
    savemat('MUJACN_LP.mat', mdict={'MUJACN_LP' : MUJACN})


if jac_agent.jac == True:
    

    CJAC = []
    AJAC = []
    MUJAC =[]
    for i in range(params['T_cycle']):
        CJAC.append((CHist[i]-C_dx0)/jac_agent.dx)
        AJAC.append((AHist[i]-A_dx0)/jac_agent.dx)
        MUJAC.append((MUHist[i]-MU_dx0)/jac_agent.dx)
        
    savemat('CJAC_LP.mat', mdict={'CJAC_LP' : CJAC})
    savemat('AJAC_LP.mat', mdict={'AJAC_LP' : AJAC})
    savemat('MUJAC_LP.mat', mdict={'MUJAC_LP' : MUJAC})

plt.plot(MU_dx0)
plt.plot((MUHist[0][1:]), label = '0')
#plt.plot((CHist[4]- C_dx0)/(jac_agent.dx), label = '175')
#plt.plot((MUHist[3][1:]), label = '100')
plt.plot((MUHist[1][1:]), label = '20')
plt.plot((MUHist[2][1:] ), label = '50')
plt.show()


plt.plot((MUHist[0]- MU_dx0)/(jac_agent.dx), label = '0')
#plt.plot((MUHist[3]- MU_dx0)/(jac_agent.dx), label = '100')
plt.plot((MUHist[1]- MU_dx0)/(jac_agent.dx), label = '20')
plt.plot((MUHist[2] - MU_dx0)/(jac_agent.dx), label = '50')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.legend()
plt.show()


plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[0], label = '0')
plt.plot(CHist[1], label = '20')
plt.plot(CHist[2], label = '50')
plt.title("Aggregate Consumption")
plt.ylabel("Aggregate Consumption")
plt.xlabel("Period")
plt.ylim([.9,1.2])
plt.legend()
#plt.savefig("AggregateConsumption.jpg", dpi=500)
plt.show()



plt.plot((CHist[0]- C_dx0)/(jac_agent.dx), label = '0')
plt.plot((CHist[1]- C_dx0)/(jac_agent.dx), label = ' 20')
plt.plot((CHist[2]- C_dx0)/(jac_agent.dx), label = ' 50')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.ylabel("dC / dw")
plt.xlabel("Period")
plt.title("Consumption Jacobians")
plt.legend()
#plt.savefig("ConsumptionJacobianComparison.jpg", dpi=500)
plt.show()


'''

plt.plot((CHist[0]- C_dx0)/(jac_agent.dx),color='darkorange', label = 'dx = .09')
#plt.plot((CHist[4]- C_dx0)/(jac_agent.dx), label = '175')
#plt.plot((CHist[3]- C_dx0)/(jac_agent.dx), label = '100')
#plt.plot((CHist[1]- C_dx0)/(jac_agent.dx), label = '20', color='gray')
plt.plot((CHist[2] - C_dx0)/(jac_agent.dx), color='darkorange')
plt.plot(CJACW.T[0],color = 'royalblue', label = 'dx =2')
plt.plot(CJACW.T[50], color = 'royalblue')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.ylabel("dC / dw")
plt.xlabel("Period")
plt.title("Consumption Jacobians")
plt.legend()
#plt.savefig("ConsumptionJacobianComparison.jpg", dpi=500)
plt.show()
'''



plt.plot(M_dx0, label = 'm steady state')
plt.plot(MHist[1], label = '20')
plt.plot(MHist[3], label = '100')
plt.plot(MHist[2], label = '50')
plt.ylabel("Agregate Wealth")
plt.xlabel("Period")
plt.title("Path of Aggregate Wealth")
plt.ylim([1,4])
plt.legend()
#plt.savefig("Path_Aggregate Wealth.jpg", dpi=500)
plt.show()


plt.plot(MHist[1] - M_dx0, label = '20')
plt.plot(MHist[0] - M_dx0, label = '0')
plt.plot(MHist[2] - M_dx0, label = '50')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.xlabel("Period")
plt.ylabel("Aggregate Wealth")
plt.title("Wealth Jacobians")
plt.legend()
#plt.savefig("Aggregate Wealth.jpg", dpi=500)
plt.show()




plt.plot(A_dx0, label = 'Asset steady state')
plt.plot(AHist[2], label = '50')
plt.plot(AHist[1], label = '20')
plt.plot(AHist[3], label = '100')
plt.plot(AHist[0], label = '0')
plt.xlabel("Period")
plt.ylabel("Aggregate Assets")
plt.title("Aggregate Assets")
plt.legend()
#plt.savefig("Aggregate Assets.jpg", dpi=500)
plt.show()




plt.plot((AHist[0]- A_dx0)/(jac_agent.dx), label = '0')
plt.plot((AHist[1]- A_dx0)/(jac_agent.dx), label = '20')
plt.plot((AHist[2] - A_dx0)/(jac_agent.dx), label = '50')
#plt.plot((AHist[3]- A_dx0)/(jac_agent.dx), label = '100')
plt.ylim([-.3,5])
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.xlabel("Period")
plt.ylabel("dA / dw")
plt.title("Asset Jacobians")
plt.legend()
#plt.savefig("AssetJacobianPerm.jpg", dpi=500)
plt.show()




# =============================================================================
 
'''


funcs=[]
list_mLvl = []
list_mNrm = []
for i in range(num_consumer_types):
    list_mLvl.append(consumers_ss[i].state_now['mNrm']*consumers_ss[i].state_now['pLvl'] )
    list_mNrm.append(consumers_ss[i].state_now['mNrm'])
    funcs.append(consumers_ss[i].solution[0].cFunc)

mNrm = np.concatenate(list_mNrm)   
mLvl = np.concatenate(list_mLvl)
plot_funcs(funcs,0,1.4)
plt.hist(mNrm, bins=np.linspace(0,1.4,num=1000))
plt.show()

plt.hist(mLvl, bins=np.linspace(0,1.2,num=1000))
plt.show()



x = np.linspace(0, 1.4, 1000, endpoint=True)

y=[]
for i in range(num_consumer_types):
    y.append(funcs[i](x))


h = np.histogram(mNrm, bins=np.linspace(0,1.4,num=1000))

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Cash on Hand')
ax1.plot(x, y[0], 'm' )


ax1.plot(x, y[1], 'k' )
ax1.plot(x, y[2], 'darkorange' )
ax1.plot(x, y[3], 'forestgreen' )
ax1.plot(x, y[4], 'deepskyblue' )
ax1.plot(x, y[5], 'r' )
ax1.plot(x, y[6], 'darkslategrey' )
ax1.set_ylim((0,.23))
ax1.set_ylabel('Consumption', color='k')


ax2= ax1.twinx()
ax2.hist(mNrm, bins=np.linspace(0,1.4,num=1000),color= 'darkviolet')
ax2.set_ylim((0,1600))
ax2.set_ylabel('Number of Households', color='k')
#plt.savefig("Presentation.png", dpi=150)


mu=1.2
sigma1= .5
mean=np.log(mu)-(sigma1**2)/2

print(((np.exp(sigma1)-1)*np.exp(2*mean+sigma1**2))**.5)

print(np.exp(mean + (sigma1**2)/2))




ad = Lognormal(mu=mean, sigma=sigma1,
            seed=123,
        ).draw(100000)

plt.hist(ad,bins=np.linspace(-2,20,num=10000))

plt.hist(aLvl,bins=np.linspace(-2,20,num=10000))




G=.3
t=.15
Inc = .3
mho=.03

w = (1/1.015)
N = (.9*(Inc*mho) + G) / (w*t) 
r = (1.06)**.25 - 1 
q = ((1-w)*N)/r

print(N)
print(N-G)
print(q)

'''








