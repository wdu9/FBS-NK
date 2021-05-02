# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:51:38 2021

@author: William Du

python 3.8.8


"""
import numpy as np
import matplotlib.pyplot as plt
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
                
                jac,
                jacW,
                cList,
                s,
                dx,
                T_sim,
                mu_u,
                L,
                PermShkStd,
                PermShkCount,
                TranShkCount,
                TranShkStd,
                tax_rate,
                UnempPrb,
                IncUnemp,
                wage,
                N,
                SSPmu,
                
                
                ):
                 
        self.s = s 
        self.dx=dx
        self.cList = cList #need to change this to time state variable somehow
        self.jac = jac
        self.jacW = jacW
        self.mu_u = mu_u
        self.L = L
        self.PermShkStd = PermShkStd
        self.PermShkCount = PermShkCount
        self.TranShkCount = TranShkCount
        self.TranShkStd = TranShkStd
        self.tax_rate = tax_rate
        self.UnempPrb = UnempPrb
        self.IncUnemp = IncUnemp
        self.wage=wage
        self.N=N
        self.SSPmu = SSPmu
                
        
        
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
        
        
    
        
       
        if self.jacW == True:
          
          if len(self.cList) == self.T_sim - self.s  :
             
            self.wage = 1/(self.SSPmu) + self.dx
        

        else:
            
              self.wage = 1/(self.SSPmu)
        
              
        PermShkDstn_U = Lognormal(np.log(self.mu_u) - (self.L*(self.PermShkStd[0])**2)/2 , self.L*self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when unemployed
        PermShkDstn_E = MeanOneLogNormal( self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when employed
        
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
        
        IncShkDstn = DiscreteDistribution(pmf_I, X_I)
        
        
        self.IncShkDstn = IncShkDstn
 
    
 
    
              
  
        
    
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
        
        if self.jac == True:
            
            if len(self.cList) == self.T_sim - self.s  :
                self.Rfree = self.Rfree + self.dx
            else:
                self.Rfree = self.Rfree
                
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
                                                   "N",
                                                   "B",
                                                   "cList",
                                                   "s",
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "jacW",
                                                   "PermShkStd",
                                                   
                                                    "PermShkCount",
                                                    "TranShkCount",
                                                    "TranShkStd",
                                                    "tax_rate",
                                                    "UnempPrb",
                                                    "IncUnemp",
                                                    "G",
                                                 
                                                   
                    
                                                  ]
    
    

    
    def __init__(self, cycles= 200, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 200, **kwds)
        
        #Steady State values for Wage , Labor and tax rate
        self.cList = []
        self.s = self.s 
        self.dx=self.dx
        
        
        solver = FBSNK_solver
        self.solve_one_period = make_one_period_oo_solver(solver)
    
    

    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate) + self.B*(self.Rfree - 1) #calculate SS labor supply from Budget Constraint
        
        
        #self.wage = .833333
        #self.N = .5
        
        
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
        
        IncShkDstnw = [DiscreteDistribution(pmf_I, X_I)]
        
        self.IncShkDstnw = IncShkDstnw
        self.add_to_time_vary('IncShkDstnw')
        
        
        
    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        
        if self.jacW == True:
            if self.t_sim == self.s:
                self.IncShkDstn = self.IncShkDstnw
                
            else:
                self.IncShkDstn = self.IncShkDstnN
                
        else:
            
            self.IncShkDstn = self.IncShkDstnN
            
        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncShkDstnNow = self.IncShkDstn[
                    t - 1
                ]  # set current income distribution
                
                
                PermGroFacNow = self.PermGroFac[t - 1]  # and permanent growth factor
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstnNow.draw(N)
                
                
                PermShkNow[these] = (
                    IncShks[0, :] * PermGroFacNow
                )  # permanent "shock" includes expected growth
                TranShkNow[these] = IncShks[1, :]
                
        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstnNow = self.IncShkDstn[0]  # set current income distribution
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstnNow.draw_events(N)
            PermShkNow[these] = (
                IncShkDstnNow.X[0][EventDraws] * PermGroFacNow
            )  # permanent "shock" includes expected growth
            TranShkNow[these] = IncShkDstnNow.X[1][EventDraws]
        #        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0
        
        
        
        # Store the shocks in self
        self.EmpNow = np.ones(self.AgentCount, dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.shocks['PermShk'] = PermShkNow
        self.shocks['TranShk'] = TranShkNow
            
        
        

                
      
        
    
    
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
        
        if self.jac == True:
            if self.t_sim == self.s  :
                RfreeNow = (self.Rfree + self.dx)* np.ones(self.AgentCount)
            else:
                RfreeNow = self.Rfree * np.ones(self.AgentCount)
        else:
            RfreeNow = self.Rfree * np.ones(self.AgentCount)
            
            
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
        
        
        if self.jac == True or self.jacW == True :
        
            if self.t_sim == 0:
                
                for i in range(num_consumer_types):
                    if  self.DiscFac == consumers_ss[i].DiscFac:
                        #mNrmNow = consumers_ss[i].state_now['mNrm']
                        #pLvlNow = list_pLvl[i]
                        mNrmNow = consumers_ss[i].history['mNrm'][self.T_sim-1,:]
                        pLvlNow = consumers_ss[i].history['pLvl'][self.T_sim-1,:]
                        print(self.DiscFac)
    

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None
    
    
    
    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.04**.25,                       # Interest factor on assets
    "DiscFac": 0.978,                     # Intertemporal discount factor
    "LivPrb" : [.9725],                   # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.08,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.1,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "T_sim" : 500,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.1, 
     "s"          : 1,
     "dx"         : .1,                  #Deviation from steady state
     "jac"        : True,
     "jacW"       : True, 
     
    #New Economy Parameters
     "SSWmu " : 1.025 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.025,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0,                               # Net Bond Supply
     "G" : .01
     }


    
C_SS = .1294 

NumAgents = 150000

###############################################################################
###############################################################################

target = 0.34505832912738216

tolerance = .01

completed_loops=0

go = True

ss_agent = FBSNK_agent(**IdiosyncDict)
ss_agent.cycles = 0
ss_agent.jac = False
ss_agent.jacW = False
ss_agent.dx = 0
ss_agent.T_sim = 1400
ss_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']


num_consumer_types = 7     # num of types 


center = 0.9875
    

while go:
    
    discFacDispersion = 0.0049
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
        consumers_ss[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
    
    

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
    aNrm = np.concatenate(list_aNrm)
    c = np.concatenate(litc)
    a = np.concatenate(list_aLvl)
    AggA = np.mean(np.array(a))
    AggC = np.mean(np.array(c))

    
    
    if AggA - target > 0 :
        
       center = center - .001
        
    elif AggA - target < 0: 
        center = center + .001
        
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
    
    go = distance >= tolerance and completed_loops < 1
        

print(AggA)
print(AggC)

################################################################################
################################################################################

ghost_agent = FBSNK_agent(**IdiosyncDict)
ghost_agent.T_sim = 200
ghost_agent.cycles = ghost_agent.T_sim
ghost_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']
ghost_agent.dx = 0
ghost_agent.jac = False
ghost_agent.jacW = True




ghosts= [] 

for i in range(num_consumer_types):
    ghosts.append(deepcopy(ghost_agent))

# now create types with different disc factors
for i in range(num_consumer_types):
        ghosts[i].DiscFac   = DiscFac_list[i]
        ghosts[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
        ghosts[i].solution_terminal = deepcopy(consumers_ss[i].solution[0]) ### Should it have a Deepcopy?
        
#############################################################################      
     

    
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
        
    Cg = np.concatenate(litc_g)
    Cg = np.mean(np.array(Cg))

    listC_g.append(Cg)
        
C_dx0 = np.array(listC_g)
         
plt.plot(C_dx0, label = 'steady state')
plt.legend()
plt.show()


###############################################################################
###############################################################################


jac_agent = FBSNK_agent(**IdiosyncDict)
jac_agent.dx = 0.8
jac_agent.jac = False
jac_agent.jacW = True

jac_agent.T_sim = 200
jac_agent.cycles = jac_agent.T_sim
jac_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']


consumers = [] 

# now create types with different disc factors

for i in range(num_consumer_types):
    consumers.append(deepcopy(jac_agent))


for i in range(num_consumer_types):
        consumers[i].DiscFac    = DiscFac_list[i]
        consumers[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
        consumers[i].solution_terminal = deepcopy(consumers_ss[i].solution[0]) ### Should it have a Deepcopy?


##############################################################################

testSet= [0,1,15,40,100]

Mega_list =[]
CHist = []
AHist=[]
MHist=[]
#for i in range(jac_agent.T_sim):
for i in testSet:

        listC = []
        listH = []
        listA = []
        listM = []
        for k in range(num_consumer_types):
            consumers[k].cList=[]
            consumers[k].s = i 
            consumers[k].solve()
            consumers[k].initialize_sim()
            consumers[k].simulate()
            
            listH.append([consumers[k].history['cNrm'],consumers[k].history['pLvl']])
            #listM.append(consumers[k].history['mNrm'])
            #listA.append([consumers[k].history['aNrm'],consumers[k].history['pLvl']])
            

            
        for j in range(jac_agent.T_sim):
            
            litc_jac= []
            lita_jac =[]
            litm_jac =[]
            for n in range(num_consumer_types):
                litc_jac.append(listH[n][0][j,:]*listH[n][1][j,:])
                #lita_jac.append(listA[n][0][j,:]*listA[n][1][j,:])
                #litm_jac.append(listM[n][j,:]*listH[n][1][j,:])

            
            c = np.concatenate(litc_jac)
            c = np.mean(np.array(c))
            listC.append(c)
            
            #a = np.concatenate(lita_jac)
            #a = np.mean(np.array(a))
            #listA.append(a)
            
            
            #m = np.concatenate(litm_jac)
            #m = np.mean(np.array(m))
            #listM.append(m)
            

        
        #AHist.append(np.array(listA))
        #MHist.append(np.array(listM))
        CHist.append(np.array(listC))
        #Mega_list.append(np.array(listC)- C_dx0)  # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s
        print(i)



plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[1], label = '1')
#plt.plot(CHist[3], label = '40')
plt.plot(CHist[2], label = '15')
plt.plot(CHist[4], label = '100')

plt.ylim([0.125,.135])
plt.legend()
plt.show()





#plt.plot((CHist[0][1:]- C_dx0[1:])/(jac_agent.dx), label = '0')
#plt.plot((CHist[3][1:]- C_dx0[1:])/(jac_agent.dx), label = '40')
plt.plot((CHist[1][1:]- C_dx0[1:])/(jac_agent.dx), label = '1')
plt.plot((CHist[2][1:] - C_dx0[1:])/(jac_agent.dx), label = '15')
plt.plot((CHist[4][1:] - C_dx0[1:])/(jac_agent.dx), label = '100')

plt.ylim([-.02,.02])
plt.legend()
plt.show()

'''
#plt.plot(C_dx0 , label = 'Steady State')
plt.plot(MHist[1], label = '1')
#plt.plot(MHist[3], label = '40')
plt.plot(MHist[2], label = '15')
plt.legend()
plt.show()
'''



'''

plt.plot(AHist[3], label = '40')
plt.plot(AHist[1], label = '3')
plt.plot(AHist[0], label = '0')

plt.legend()
plt.show()



plt.ylim([0.036,.039])
plt.legend()
plt.show()

#plt.plot(CHist[56], label = '56')
plt.plot(CHist[44], label = '44')

#plt.plot(CHist[49], label = '49')
plt.plot(CHist[35], label = '35')

plt.plot(CHist[0], label = '0')

plt.ylim([0.035,.045])
plt.legend()
plt.show()



'''
G=.01
t=.1
Inc = .08
mho=.05

w = (1/1.025)
N = (.9*(Inc*mho)+G)/ (w*t) 
r = (1.04)**.25 -1
print(N)

N1 = (.9*(.08*.05)+.01)/ (w*t) 


q = ((1-w)*N1)/r

print(N1)
print(N1-N)
print(q)


