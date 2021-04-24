# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:51:38 2021

@author: wdu
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
from scipy.io import loadmat

from HARK.utilities import plot_funcs_der, plot_funcs


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





class FBSNK_agent(IndShockConsumerType):
    
   
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
        self.Rfree = 1.03
        self.cList = []
        self.s = self.s 
        self.dx=self.dx
        
        
        solver = FBSNK_solver
        self.solve_one_period = make_one_period_oo_solver(solver)
    

    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ))/ (self.wage*self.tax_rate) + self.B*(self.Rfree - 1) #calculate SS labor supply from Budget Constraint
        
        
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
        
    
    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        Returns
        -------
        None
        """
        
        if self.jac == False:
            
            # Get and store states for newly born agents
            N = np.sum(which_agents)  # Number of new consumers to make
            self.state_now['aNrm'][which_agents] = Lognormal(
                mu=self.aNrmInitMean,
                sigma=self.aNrmInitStd,
                seed=self.RNG.randint(0, 2 ** 31 - 1),
            ).draw(N)
            # why is a now variable set here? Because it's an aggregate.
            pLvlInitMeanNow = self.pLvlInitMean + np.log(
                self.state_now['PlvlAgg']
            )  # Account for newer cohorts having higher permanent income
            self.state_now['pLvl'][which_agents] = Lognormal(
                pLvlInitMeanNow,
                self.pLvlInitStd,
                seed=self.RNG.randint(0, 2 ** 31 - 1)
            ).draw(N)
        
        else: 
                for i in range(num_consumer_types):
                    
                    if self.DiscFac == consumers_ss[i].DiscFac:
                        
                        self.state_now['aNrm'] = list_aNrm[i]
                        self.state_now['pLvl'] = list_pLvl[i]
                        
            
        self.t_age[which_agents] = 0  # How many periods since each agent was born
        self.t_cycle[
            which_agents
        ] = 0  # Which period of the cycle each agent is currently in
        return None
    
    
    
    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2.0,                           # Coefficient of relative risk aversion
    #"Rfree": 1.03,                       # Interest factor on assets
    "DiscFac": 0.978,                     # Intertemporal discount factor
    "LivPrb" : [.97],                   # Survival probability
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
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     "s"          : 1,
     "dx"         : .0001,                  #Deviation from steady state
     "jac"        : True,
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.2,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0                               # Net Bond Supply
    
}


    
###############################################################################

###############################################################################
###############################################################################

target = .6

tolerance = .01

completed_loops=0

go = True

ss_agent = FBSNK_agent(**IdiosyncDict)
ss_agent.cycles=0
ss_agent.jac = False
ss_agent.dx = 0
ss_agent.T_sim = 300
ss_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']


num_consumer_types = 7     # num of types 

center = 0.97974
    

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
    
    print('Completed loops')
    print(completed_loops)
    
    go = distance >= tolerance and completed_loops < 100
        

print(AggA)
print(AggC)

################################################################################
################################################################################

ghost_agent = FBSNK_agent(**IdiosyncDict)
ghost_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']
ghost_agent.dx = 0
ghost_agent.jac = True


num_consumer_types = 7     # num of types 

center = 0.97974

    
discFacDispersion = 0.0069
bottomDiscFac     = center - discFacDispersion
topDiscFac        = center + discFacDispersion

DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
DiscFac_list  = DiscFac_dist.X


ghosts= [] 

for i in range(num_consumer_types):
    ghosts.append(deepcopy(ghost_agent))

# now create types with different disc factors
for i in range(num_consumer_types):
        ghosts[i].DiscFac   = DiscFac_list[i]
        ghosts[i].AgentCount = int(100000*DiscFac_dist.pmf[i])
        ghosts[i].solution_terminal = consumers_ss[i].solution[0]
        
        
listC_g = []
listH_g = []
    
for k in range(num_consumer_types):
    ghosts[k].solve()
    ghosts[k].initialize_sim()
    ghosts[k].simulate()

    listH_g.append([ghosts[k].history['cNrm'], ghosts[k].history['pLvl']])

for j in range(ghost_agent.T_sim):

    litc_g=[]
    for n in range(num_consumer_types):
        litc_g.append(listH_g[n][0][j,:]*listH_g[n][1][j,:])
        
    Cg = np.concatenate(litc)
    Cg = np.mean(np.array(Cg))

    listC_g.append(Cg)
        
C_dx0 = np.array(listC_g)
    
    
###############################################################################
###############################################################################


jac_agent = FBSNK_agent(**IdiosyncDict)
jac_agent.jac = True
jac_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']

num_consumer_types = 7     # num of types 

center = 0.97974

discFacDispersion = 0.0069
bottomDiscFac     = center - discFacDispersion
topDiscFac        = center + discFacDispersion

DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
DiscFac_list  = DiscFac_dist.X


consumers = [] 

# now create types with different disc factors

for i in range(num_consumer_types):
    consumers.append(deepcopy(jac_agent))


for i in range(num_consumer_types):
        consumers[i].DiscFac    = DiscFac_list[i]
        consumers[i].AgentCount = int(100000*DiscFac_dist.pmf[i])
        consumers[i].solution_terminal = consumers_ss[i].solution[0]

        
##############################################################################

Mega_list =[]

for i in range(jac_agent.T_sim):
    
        listC = []
        listH = []
        for k in range(num_consumer_types):
            consumers[k].s=i
            consumers[k].solve()
            consumers[k].initialize_sim()
            consumers[k].simulate()
            
            listH.append([consumers[k].history['cNrm'],consumers[k].history['pLvl']])
            
        for j in range(jac_agent.T_sim):
            
            litc=[]
            for n in range(num_consumer_types):
                litc.append(listH[n][0][j,:]*listH[n][1][j,:])
            
            c = np.concatenate(litc)
            c = np.mean(np.array(c))
        
            listC.append(c)
        
        Mega_list.append(np.array(listC)- C_dx0) # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s
        
        print(i)

















    
    
        
        