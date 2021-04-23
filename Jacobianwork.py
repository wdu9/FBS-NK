# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:49:00 2021

@author: William Du


Python Version 3.8.8

HARK version 11.0
"""

import numpy as np
from copy import copy, deepcopy
from timeit import timeit
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform
from HARK.utilities import get_percentiles, get_lorenz_shares, calc_subpop_avg
from HARK import Market, make_one_period_oo_solver

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




##############################################################################
##############################################################################
class FBSNK_solver(ConsIndShockSolver):
    
   
    def __init__(self, 
                solution_next,
                IncShkDstn,
                LivPrb,
                DiscFac,
                CRRA,
                Rfree,
                PermGroFac,
                BoroCnstArt,
                aXtraGrid,
                vFuncBool,
                CubicBool,
                
                cList,
                s,
                dx,
                T_sim,
                ):
                 
        self.s = s 
        self.dx=dx
        self.cList = cList #need to change this to time state variable somehow
        
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool
        self.def_utility_funcs()
        self.Rfree=Rfree
        self.T_sim = T_sim
        
       
        
        if len(self.cList) == self.T_sim - 1  - self.s  :
            self.Rfree = Rfree + self.dx
        else:
            self.Rfree = Rfree
            

        
    
    def solve(self):
        """
        Solves the single period consumption-saving problem using the method of
        endogenous gridpoints.  Solution includes a consumption function cFunc
        (using cubic or linear splines), a marginal value function vPfunc, a min-
        imum acceptable level of normalized market resources mNrmMin, normalized
        human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  It might also
        have a value function vFunc and marginal marginal value function vPPfunc.
        Parameters
        ----------
        none
        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        # Make arrays of end-of-period assets and end-of-period marginal value
        aNrm = self.prepare_to_calc_EndOfPrdvP()
        EndOfPrdvP = self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        if self.CubicBool:
            solution = self.make_basic_solution(
                EndOfPrdvP, aNrm, interpolator=self.make_cubic_cFunc
            )
        else:
            solution = self.make_basic_solution(
                EndOfPrdvP, aNrm, interpolator=self.make_linear_cFunc
            )

        solution = self.add_MPC_and_human_wealth(solution)  # add a few things
        solution = self.add_stable_points(solution)
            
        
        
        self.cList.append(solution.cFunc)
        
        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used (to prepare for next period)
        if self.vFuncBool:
            solution = self.add_vFunc(solution, EndOfPrdvP)
        if self.CubicBool:
            solution = self.add_vPPfunc(solution)
        return solution
    





##############################################################################
##############################################################################





class FBSNKagent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                   "L",
                                                   "SSPmu",
                                                   "wage",
                                                   "B",
                                                   "cList",
                                                   "s",
                                                   "dx",
                                                   "T_sim"
                                                   
                    
                                                  ]
    
    

    
    def __init__(self, cycles= 200, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 200, **kwds)
        
        #Steady State values for Wage , Labor and tax rate
        self.Rfree = 1.02
        self.cList = []
        self.s = self.s 
        self.dx=self.dx
        
        
        solver = FBSNK_solver
        self.solve_one_period = make_one_period_oo_solver(solver)
    

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
    
        
        if self.t_sim == self.s:
            RfreeNow = (self.Rfree + self.dx)* np.ones(self.AgentCount)
        else:
            RfreeNow = self.Rfree * np.ones(self.AgentCount)
            
        return RfreeNow
        
        
        
    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """   
        
            
    
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        
        
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
                self.state_now['mNrm'][these]
            )
        self.controls['cNrm'] = cNrmNow

        # MPCnow is not really a control
        self.MPCnow = MPCnow
        return None

    
    
    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2.0,                           # Coefficient of relative risk aversion
    #"Rfree": 1.03,                       # Interest factor on assets
    "DiscFac": 0.978,                     # Intertemporal discount factor
    "LivPrb" : [.9745],                   # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

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
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .8 ,
     "L"          : 1.3, 
     "s"          : 1,
     "dx"         : .0001,                   #Deviation from steady state
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.2,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0                               # Net Bond Supply
    
}


    
###############################################################################
###############################################################################




example1 = FBSNKagent(**IdiosyncDict)
example1.track_vars = ['aNrm','mNrm','cNrm','pLvl']
example1.dx = 0
#example.solution_terminal = 


num_consumer_types = 7     # num of types 

center = 0.978818264

    
discFacDispersion = 0.0069
bottomDiscFac     = center - discFacDispersion
topDiscFac        = center + discFacDispersion

DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
DiscFac_list  = DiscFac_dist.X


consumers = [] 

# now create types with different disc factors
for i in range(num_consumer_types):
        example1.DiscFac    = DiscFac_list[i]
        example1.AgentCount = int(100000*DiscFac_dist.pmf[i])
        consumers.append(example1)
        
'''
#lita=[]
litc=[]
# simulate and keep track mNrm and MPCnow
for i in range(num_consumer_types):
    #consumers[i].Rfree = ss_agent.Rfree 
    consumers[i].solve()
    consumers[i].initialize_sim()
    consumers[i].simulate()
    
    
    for j in range(num_consumer_types)

    litc.append(consumers[i].history['cNrm'][,:]*consumers[i].history['pLvl'][199,:])
    #lita.append(consumers[i].state_now['aLvl'])
    
    print('one consumer solved and simulated')
    
    c = np.concatenate(litc)
    #a = np.concatenate(lita)
    #A_ss = np.mean(np.array(a))
    C_ss = np.mean(np.array(c))
'''

listC = []
listH = []
for k in range(num_consumer_types):
    consumers[k].s=i
    consumers[k].solve()
    consumers[k].initialize_sim()
    consumers[k].simulate()

    listH.append([consumers[k].history['cNrm'],consumers[k].history['pLvl']])

    for j in range(example1.T_sim):

        litc=[]
        for n in range(num_consumer_types):
            litc.append(listH[n][0][j,:]*listH[n][1][j,:])
    
        c = np.concatenate(litc)
        c = np.mean(np.array(c))

        listC.append(c)
        
    C_dx0 = np.array(listC)
        
        



###############################################################################


example2 = FBSNKagent(**IdiosyncDict)
example2.track_vars = ['aNrm','mNrm','cNrm','pLvl']


######################################################################################
num_consumer_types = 7     # num of types 

center = 0.978818264

discFacDispersion = 0.0069
bottomDiscFac     = center - discFacDispersion
topDiscFac        = center + discFacDispersion

DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
DiscFac_list  = DiscFac_dist.X


consumers = [] 

# now create types with different disc factors
for i in range(num_consumer_types):
        
        example2.DiscFac    = DiscFac_list[i]
        example2.AgentCount = int(10000*DiscFac_dist.pmf[i])
        consumers.append(example2)

##############################################################################

Mega_list =[]

for i in range(example2.T_sim):
    
        listC = []
        listH = []
        for k in range(num_consumer_types):
            consumers[k].s=i
            consumers[k].solve()
            consumers[k].initialize_sim()
            consumers[k].simulate()
            
            listH.append([consumers[k].history['cNrm'],consumers[k].history['pLvl']])
            
        for j in range(example2.T_sim):
            
            litc=[]
            for n in range(num_consumer_types):
                litc.append(listH[n][0][j,:]*listH[n][1][j,:])
            
            c = np.concatenate(litc)
            c = np.mean(np.array(c))
        
            listC.append(c)
        
        Mega_list.append(np.array(listC) - C_dx0) # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s
        
        print(i)
    
    
''' 
    list_c = []
    for j in range(example2.T_sim):
        c = np.array((example2.history['cNrm'][j,:]*example2.history['pLvl'][j,:]))
        c = np.mean(c)
        list_c.append(c- 0.06467790625604977 - 0.06465157617443441)
    
    Mega_list.append(np.array(list_c))
    

'''






