# -*- coding: utf-8 -*-


"""
Created on Sun Apr  4 21:41:21 2021

author: William Du

Python Version 3.8.8

HARK version 11.0
"""


import numpy as np
from copy import copy, deepcopy
from time import time
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform
from HARK.utilities import get_percentiles, get_lorenz_shares, calc_subpop_avg
from HARK import Market
import HARK.ConsumptionSaving.ConsIndShockModel as ConsIndShockModel
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    IndShockConsumerType,
    PerfForesightConsumerType,
)
from HARK.distribution import Uniform

from HARK.ConsumptionSaving.ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType
from HARK import MetricObject, Market, AgentType
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from HARK.utilities import plot_funcs_der, plot_funcs




##############################################################################
##############################################################################





class FBSNK_ss_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                   "L",
                                                   "SSPmu",
                                                   "wage",
                                                   "B"
                                                   
                                                  ]
 
    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles= 0, **kwds)
        
        #Steady State values for Wage , Labor and tax rate
        self.Rfree = 1.03
        #MVMU = wage*(1-self.tax_rate)/(self.SSWmu)
        
    

    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ))/ (self.wage*self.tax_rate) + self.B*(self.Rfree -1) #calculate SS labor supply from Budget Constraint
        
        
        PermShkDstn_U = Lognormal(np.log(self.mu_u) - (self.L*(self.PermShkStd[0])**2)/2 , self.L*self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when unemployed
        PermShkDstn_E = MeanOneLogNormal( self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when employed
        
        
        pmf_P = np.concatenate(((1-self.UnempPrb)*PermShkDstn_E.pmf ,self.UnempPrb*PermShkDstn_U.pmf)) 
        X_P = np.concatenate((PermShkDstn_E.X, PermShkDstn_U.X))
        PermShkDstn = [DiscreteDistribution(pmf_P, X_P)]
        self.PermShkDstn = PermShkDstn 
        
        TranShkDstn_E = MeanOneLogNormal( self.TranShkStd[0],123).approx(self.TranShkCount)#Transitory Shock Distribution faced when employed
        TranShkDstn_E.X = (TranShkDstn_E.X *(1-self.tax_rate)*self.wage*self.N)/(1-self.UnempPrb)**2 #add wage, tax rate and labor supply
        
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
        self.IncShkDstn = IncShkDstn
        self.add_to_time_vary('IncShkDstn')
    
    
    
    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2.0,                           # Coefficient of relative risk aversion
    #"Rfree": 1.03,                         # Interest factor on assets
    "DiscFac": 0.978,                       # Intertemporal discount factor
    "LivPrb" : [.97],                     # Survival probability
    "PermGroFac" :[1.00],                  # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [(0.01*4)**0.5],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.15,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other parameters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type

    # Parameters only used in simulation
    "AgentCount" : 100000,                  # Number of agents of this type
    "T_sim" : 300,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.2,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0                               # Net Bond Supply
    
}


    
###############################################################################
'''

example0 = FBSNKagent(**IdiosyncDict)
example0.cycles=0
example0.solve()


example1 = FBSNKagent(**IdiosyncDict)
example1.solve()

print(example1.solution[0].cFunc.functions[0].y_list -example0.solution[0].cFunc.functions[0].y_list)

'''
###############################################################################
###############################################################################

target = .6

tolerance = .0001

completed_loops=0

go = True

ss_agent = FBSNK_ss_agent(**IdiosyncDict)
ss_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']


num_consumer_types = 7     # num of types 

center = 0.97975
    

while go:
    
    discFacDispersion = 0.0069
    bottomDiscFac     = center - discFacDispersion
    topDiscFac        = center + discFacDispersion
    
    DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
    DiscFac_list  = DiscFac_dist.X
    
    consumers_ss = [] 
    
    # now create types with different disc factors
    for i in range(num_consumer_types):
        consumers_ss.append(deepcopy(ss_agent))
        
    for i in range(num_consumer_types):
        consumers_ss[i].DiscFac    = DiscFac_list[i]
        consumers_ss[i].AgentCount = int(100000*DiscFac_dist.pmf[i])
    
    

    list_pLvl=[]
    list_aNrm =[]
    list_aLvl=[]
    litc=[]
    # simulate and keep track mNrm and MPCnow
    for i in range(num_consumer_types):
        consumers_ss[i].solve()
        consumers_ss[i].initialize_sim()
        consumers_ss[i].simulate()
        
        list_pLvl.append(consumers_ss[i].state_now['pLvl'])
        list_aNrm.append(consumers_ss[i].state_now['aNrm'])
        litc.append((consumers_ss[i].state_now['mNrm'] - consumers_ss[i].state_now['aNrm'])*consumers_ss[i].state_now['pLvl'])
        list_aLvl.append(consumers_ss[i].state_now['aLvl'])
        
        print('one consumer solved and simulated')
    
    pLvl = np.concatenate(list_pLvl)
    aNrm = np.concatenate(list_aLvl)
    c = np.concatenate(litc)
    a = np.concatenate(list_aLvl)
    AggA = np.mean(np.array(a))
    AggC = np.mean(np.array(c))

    
    
    if AggA - target > 0 :
        
       center = center - .00001
        
    elif AggA - target < 0: 
        center = center + .00001
        
    else:
        break
    
    print('Assets')
    print(AggA)
    print('consumption')
    print(AggC)
    print('center')
    print(center)
    
    distance = abs(AggA - target) 
    
    completed_loops += 1
    
    print('Completed loops')
    print(completed_loops)
    
    go = distance >= tolerance and completed_loops < 100
        

print(AggA)
print(AggC)




##############################################################################
################################################################################
################################################################################

'''

num_consumer_types = 7     # num of types 

center = 0.9787
discFacDispersion = 0.0069
bottomDiscFac     = center - discFacDispersion
topDiscFac        = center + discFacDispersion

DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
DiscFac_list  = DiscFac_dist.X



target = .75

tolerance = .001

completed_loops=0

go = True

example = FBSNK_ss_agent(**IdiosyncDict)

char_view_consumers = [] 
    
# now create types with different disc factors
for i in range(num_consumer_types):
    example.DiscFac    = DiscFac_list[i]
    example.AgentCount = int(10000*DiscFac_dist.pmf[i])
    char_view_consumers.append(example)
   

while go:

    lita=[]
    litc=[]
    # simulate and keep track mNrm and MPCnow
    for i in range(num_consumer_types):
        char_view_consumers[i].Rfree = example.Rfree 
        char_view_consumers[i].solve()
        char_view_consumers[i].initialize_sim()
        char_view_consumers[i].simulate()
        
        litc.append((char_view_consumers[i].state_now['mNrm'] - char_view_consumers[i].state_now['aNrm'])*char_view_consumers[i].state_now['pLvl'])
        lita.append(char_view_consumers[i].state_now['aLvl'])
        print('k')
    
    c = np.concatenate(litc)
    a = np.concatenate(lita)
    AggA = np.mean(np.array(a))
    AggC = np.mean(np.array(c))

    
    
    if AggA - target > 0 :
        
       example.Rfree = example.Rfree - .001
        
    elif AggA - target < 0: 
        example.Rfree = example.Rfree + .001
        
    else:
        break
    
    print('Assets')
    print(AggA)
    print('consumption')
    print(AggC)
    print('interest rate')
    print(example.Rfree)
    
    distance = abs(AggA - target) 
    
    completed_loops += 1
    go = distance >= tolerance and completed_loops < 100
        

print(AggA)
print(AggC)

  
'''

##########################################################################
##########################################################################

###############################################################################################





