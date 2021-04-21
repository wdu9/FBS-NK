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





class FBSNKagent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                   "L",
                                                   "SSPmu",
                                                   "wage",
                                                   "B"
                                                   
                                                  ]
 
    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles= 0, **kwds)
        
        #Steady State values for Wage , Labor and tax rate
        self.Rfree = 1.02
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
    "LivPrb" : [.9745],                     # Survival probability
    "PermGroFac" :[1.00],                  # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],      # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [(0.01*4)**0.5],          # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.2,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "AgentCount" : 10000,                  # Number of agents of this type
    "T_sim" : 350,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .8 ,
     "L"          : 1.3, 
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.2,                        # Price Markup from sequence space jacobian appendix
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

target = .75

tolerance = .001

completed_loops=0

go = True

example = FBSNKagent(**IdiosyncDict)

num_consumer_types = 7     # num of types 

center = 0.9787
    

while go:
    
    discFacDispersion = 0.0069
    bottomDiscFac     = center - discFacDispersion
    topDiscFac        = center + discFacDispersion
    
    DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
    DiscFac_list  = DiscFac_dist.X
    
    char_view_consumers = [] 
    
    # now create types with different disc factors
    for i in range(num_consumer_types):
        example.DiscFac    = DiscFac_list[i]
        example.AgentCount = int(10000*DiscFac_dist.pmf[i])
        char_view_consumers.append(example)
       

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
        
       center = center - .0001
        
    elif AggA - target < 0: 
        center = center + .0001
        
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
    go = distance >= tolerance and completed_loops < 100
        

print(AggA)
print(AggC)

  




##############################################################################
########################################################################################

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

example = FBSNKagent(**IdiosyncDict)

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
'''

tolerance = .001

completed_loops=0

go = True

example = FBSNKagent(**IdiosyncDict)

while go:
    
    example.solve()

#plot_funcs(example.solution[0].cFunc,example.solution[0].mNrmMin,5)
    example.track_vars = ['aNrm','mNrm','cNrm','pLvl']
    example.initialize_sim()
    example.simulate()

    a = example.state_now['aLvl']
    AggA = np.mean(np.array(a))
    
   
    
    
    if AggA - .75 > 0 :
        
       example.Rfree = example.Rfree - .0001
        
    elif AggA-.75 < 0: 
        example.Rfree = example.Rfree + .0001
        
    else:
        break
    
    print(example.Rfree)
    
    distance = abs(AggA - .75) 
    
    completed_loops += 1
    go = distance >= tolerance and completed_loops < 100
        
    
a= example.state_now['aLvl']
c = (example.state_now['mNrm'] - example.state_now['aNrm'] )*example.state_now['pLvl']

AggA = np.mean(np.array(a))
AggC = np.mean(np.array(c))
print(AggA)
print(AggC)

'''
###############################################################################################

'''

CRRA = 2.0
DiscFac = 0.96

# Parameters for a Cobb-Douglas economy
PermGroFacAgg = 1.00  # Aggregate permanent income growth factor
PermShkAggCount = (
    3  # Number of points in discrete approximation to aggregate permanent shock dist
)
TranShkAggCount = (
    3  # Number of points in discrete approximation to aggregate transitory shock dist
)
PermShkAggStd = 0.0063  # Standard deviation of log aggregate permanent shocks
TranShkAggStd = 0.0031  # Standard deviation of log aggregate transitory shocks
DeprFac = 0.025  # Capital depreciation rate
CapShare = 0.36  # Capital's share of income
DiscFacPF = DiscFac  # Discount factor of perfect foresight calibration
CRRAPF = CRRA  # Coefficient of relative risk aversion of perfect foresight calibration
intercept_prev = 0.0  # Intercept of aggregate savings function
slope_prev = 1.0  # Slope of aggregate savings function
verbose_cobb_douglas = (
    True  # Whether to print solution progress to screen while solving
)
T_discard = 200  # Number of simulated "burn in" periods to discard when updating AFunc
DampingFac = 0.5  # Damping factor when updating AFunc; puts DampingFac weight on old params, rest on new
max_loops = 20  # Maximum number of AFunc updating loops to allow



EconomyDict = {
    "PermShkAggCount": PermShkAggCount,
    "TranShkAggCount": TranShkAggCount,
    "PermShkAggStd": PermShkAggStd,
    "TranShkAggStd": TranShkAggStd,
    "DeprFac": DeprFac,
    "SSWmu ": 1.1,
    "DiscFac": DiscFacPF,
    "CRRA": CRRAPF,
    "PermGroFacAgg": PermGroFacAgg,
    "AggregateL": 1.0,
    "intercept_prev": intercept_prev,
    "slope_prev": slope_prev,
    "verbose": verbose_cobb_douglas,
    "T_discard": T_discard,
    "DampingFac": DampingFac,
    "max_loops": max_loops,
    
    "SSWmu ": 1.1,
    "SSPmu":  1.2, # sequence space jacobian appendix
    " calvo price stickiness":  .926, # Auclert et al 2020
    "calvo wage stickiness": .899, #Auclert et al 2020
}





class Economy(Market):
    
    def __init__(self, agents=None, tolerance=0.0001, act_T=12, **kwds):
        agents = agents if agents is not None else list()
        params = EconomyDict.copy() # this is to add eocnomy dictionary
        
        params["sow_vars"] = [
            "Rfree",
        ]
        params.update(kwds)

        Market.__init__(
            self,
            agents=agents,
            reap_vars=['aLvl', 'pLvl'],
            track_vars=[],
            dyn_vars=["Rfree"],
            tolerance=tolerance,
            act_T=act_T,
            **params
        )
        self.update()

    def update(self):
        
        
    def solve(self):
        self.solveAgents()
        
        
        
    def mill_rule(self, aLvl):
        
        
        return self.Calc_endo
        
        
    def calc_dynamics(self,AggA):
        
 
            
        
        return Calc_Rules
    
class AggDynRule(MetricObject):
    
    def __init__(self,Rfree):
        self.Rfree = Rfree
        self.distance_criteria = ["AFunc"]

'''
