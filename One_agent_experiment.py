# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:38:40 2021

@author: wdu
"""

# Initial imports and notebook setup, click arrow to show
from HARK.utilities import plot_funcs_der, plot_funcs
import matplotlib.pyplot as plt
import numpy as np
mystr = lambda number : "{:.4f}".format(number)

from copy import copy, deepcopy
from time import time
from HARK.interpolation import (
    CubicInterp,
    LowerEnvelope,
    LinearInterp,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA
)


import HARK.ConsumptionSaving.ConsIndShockModel as ConsIndShockModel
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    IndShockConsumerType,
    PerfForesightConsumerType,
)

from HARK import Market, make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform, calc_expectation

from HARK.core import solve_one_cycle


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
                cList,
                s,
                dx,
                T_sim,
                completed,
                ):
                 
        self.s = s 
        self.dx=dx
        self.cList = cList #need to change this to time state variable somehow
        self.jac = jac
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
        self.completed = completed
        
        '''
        if self.jac == True and self.completed_cycles == self.T_sim -self.s:
            self.Rfree =  self.Rfree + self.dx
            
        
        '''
        
    def m_nrm_next_dx(self, shocks, a_nrm):
        """
        Computes normalized market resources of the next period
        from income shocks and current normalized market resources.
        Parameters
        ----------
        shocks: [float]
            Permanent and transitory income shock levels.       a_nrm: float
            Normalized market assets this period
        Returns
        -------
        float
           normalized market resources in the next period
        """
        return self.Rfree / (self.PermGroFac * shocks[0]) \
            * a_nrm + shocks[1] + .5

    def calc_EndOfPrdvP_dx(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.mNrmNext).
        Parameters
        ----------
        none
        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        def vp_next(shocks, a_nrm):
            return shocks[0] ** (-self.CRRA) \
                * self.vPfuncNext(self.m_nrm_next_dx(shocks, a_nrm))

        EndOfPrdvP = (
            self.DiscFacEff
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA)
            * calc_expectation(
                self.IncShkDstn,
                vp_next,
                self.aNrmNow
            )
        )

        return EndOfPrdvP    

        
    
    
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
        
        '''
        
        if self.jac == True and self.completed_cycles == self.T_sim - self.s:
            self.Rfree = 1.03**.25 + self.dx
            
        else:
            self.Rfree = 1.03**.25

            
        '''
    
        if self.jac==True and len(self.cList) == self.T_sim - self.s:
            self.Rfree =1.03**.25 +self.dx
        else:
            self.Rfree = 1.03**.25
        
        

     
        aNrm = self.prepare_to_calc_EndOfPrdvP()
        
        '''
    
        if self.jac==True and len(self.cList) == self.T_sim - self.s:
            
            EndOfPrdvP= self.calc_EndOfPrdvP_dx()
            
        else:
            EndOfPrdvP = self.calc_EndOfPrdvP()
            
        '''

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




class Test_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                 
                                                   "B",
                                                   "cList",
                                                   "s",
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "Ghost",
                                                   "completed",
                                                   #"wage",
                                                   #"N",
                                                  
                                                   
                                                   
                    
                                                  ]
    
    

    
    def __init__(self, cycles= 200, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 200, **kwds)
        
        self.cList = []
        self.completed = 95
        
        solver = FBSNK_solver
        self.solve_one_period = make_one_period_oo_solver(solver)
        
        
        
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ))/ (self.wage*self.tax_rate) + self.B*(self.Rfree - 1) #calculate SS labor supply from Budget Constraint
        
        self.wage=.833333
        self.N=1
        
        
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
    
        
        if self.jac==True and  self.t_sim == self.s-1  :
            RfreeNow = (1.03**.25 + self.dx)* np.ones(self.AgentCount)
        else:
            RfreeNow = 1.03**.25 * np.ones(self.AgentCount)
            
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
        
        
        if self.jac == True or self.Ghost == True:
            if self.t_sim == 0:
                
                    mNrmNow = ss.history['mNrm'][self.T_sim-1,:]
                    pLvlNow = ss.history['pLvl'][self.T_sim-1,:]
                    #mNrmNow = ss.state_now['mNrm']
                    #pLvlNow = ss.state_now['pLvl']
                    
                       

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None




    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.03**.25,                       # Interest factor on assets
    "DiscFac": 0.987,                     # Intertemporal discount factor
    "LivPrb" : [.985],                   # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.2,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.3,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "AgentCount" : 150000,                  # Number of agents of this type
    "T_sim" : 100,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(.27)-(.5**2)/2,                 # Mean of log initial assets
    "aNrmInitStd"  : .3,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     "s"          : 7,
     "dx"         : .1,                  #Deviation from steady state
     "jac"        : True,
     "Ghost"      : False,
     
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.2,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0                               # Net Bond Supply
    
}


###############################################################################

ss = Test_agent(**IdiosyncDict )
ss.jac= False 
ss.track_vars = ['aNrm','mNrm','cNrm','pLvl']
ss.cycles=0
ss.dx=0
ss.T_sim= 1200
ss.solve()
ss.initialize_sim()
ss.simulate()

###############################################################################
 
listC_g = []
listA_g = []
listM_g = []

    
ss_dx = Test_agent(**IdiosyncDict )
ss_dx.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']
ss_dx.solution_terminal = deepcopy( ss.solution[0])
ss_dx.jac = False
ss_dx.Ghost = True

ss.cList=[]
ss_dx.dx = 0
ss_dx.cycles= ss_dx.T_sim
ss_dx.solve()
ss_dx.initialize_sim()
ss_dx.simulate()


for j in range(ss_dx.T_sim):

    Mg = np.mean(ss_dx.history['mNrm'][j,:]*ss_dx.history['pLvl'][j,:])
    Ag = np.mean(ss_dx.history['aLvl'][j,:])
    Cg = np.mean(ss_dx.history['cNrm'][j,:]*ss_dx.history['pLvl'][j,:])

    listM_g.append(Mg)
    listA_g.append(Ag)
    listC_g.append(Cg)
    
M_dx0 = np.array(listM_g)
A_dx0 = np.array(listA_g)
C_dx0 = np.array(listC_g)

plt.plot(C_dx0, label="steady state")
plt.legend()
plt.show()

print('done with steady state')

##############################################################################

example = Test_agent(**IdiosyncDict )
example.solution_terminal = deepcopy(ss.solution[0])
example.T_sim = 100
example.cycles=example.T_sim
example.jac = True
example.dx= 0.1
example.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']



Test_set=[1,5,20,50]
Mega_list =[]
CHist = []
AHist =[]
MHist =[]
for i in Test_set:
    
        listM = []
        listC = []
        listA = []
        
        example.cList=[]
        example.s = i 
        example.solve()
        example.initialize_sim()
        example.simulate()
        

        for j in range(example.T_sim):
           
            m = np.mean(example.history['mNrm'][j,:]*example.history['pLvl'][j,:])
            a = np.mean(example.history['aLvl'][j,:])
            c = np.mean(example.history['cNrm'][j,:]*example.history['pLvl'][j,:])
        
            listM.append(m)
            listC.append(c)
            listA.append(a)
          
        MHist.append(np.array(listA))
        AHist.append(np.array(listA))
        CHist.append(np.array(listC))
        #Mega_list.append(np.array(listC)- C_dx0)  # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s

        print(i)
        
        
'''
plt.plot(C_dx0 , label = 'Steady State')

plt.plot(CHist[1], label = '1')
plt.plot(CHist[3], label = '20')

plt.plot(CHist[2], label = '5')
plt.plot(CHist[0], label = '0')

plt.ylim([.9,1.1])
plt.legend()
plt.show()


plt.plot(CHist[1]-C_dx0, label = '1')
plt.plot(CHist[3]-C_dx0, label = '20')
plt.plot(CHist[2]-C_dx0, label = '5')
plt.plot(CHist[0]-C_dx0, label = '0')
plt.ylim([-.01,.01])
plt.legend()
plt.show()

'''


'''
With new income process

plt.plot(C_dx0 , label = 'Steady State')

plt.plot(CHist[1], label = '1')
plt.plot(CHist[3], label = '20')
plt.plot(CHist[0], label = '0')

plt.ylim([.035,.038])
plt.legend()
plt.show()


plt.plot((CHist[1]-C_dx0)/(.1), label = '1')
plt.plot((CHist[3]-C_dx0)/(.1), label = '20')
plt.plot((CHist[2]-C_dx0)/(.1), label = '5')
plt.plot((CHist[0]-C_dx0)/(.1), label = '0')
plt.ylim([-.006,.003])
plt.legend()
plt.show()

'''


plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[1], label = '5')
plt.plot(CHist[2], label = '20')
plt.plot(CHist[0], label = '1')

plt.ylim([0.4,0.5])
plt.legend()
plt.show()


plt.plot((CHist[1]-C_dx0)/(.1), label = '5')
plt.plot((CHist[2]-C_dx0)/(.1), label = '20')
plt.plot((CHist[0]-C_dx0)/(.1), label = '1')
plt.plot(np.zeros(100), 'k')

plt.ylim([-.5,.5])
plt.legend()
plt.show()



plt.plot(M_dx0 , label = 'Steady State')
plt.plot(MHist[1], label = '5')
plt.plot(MHist[2], label = '20')
plt.plot(MHist[0], label = '1')
plt.ylim([0,4])
plt.legend()
plt.show()

plt.plot((MHist[1]-M_dx0)/(.1), label = '5')
plt.plot((MHist[2]-M_dx0)/(.1), label = '20')
plt.plot((MHist[0]-M_dx0)/(.1), label = '1')
plt.plot(np.zeros(100), 'k')

plt.ylim([-7,10])
plt.legend()
plt.show()


plt.plot(A_dx0 , label = 'Steady State')
plt.plot(AHist[1], label = '5')
plt.plot(AHist[2], label = '20')
plt.plot(AHist[0], label = '1')
plt.ylim([0,2])
plt.legend()
plt.show()

plt.plot((AHist[1][1:]-A_dx0[1:])/(.1), label = '5')
plt.plot((AHist[2][1:]-A_dx0[1:])/(.1), label = '20')
plt.plot((AHist[0][1:]-A_dx0[1:])/(.1), label = '1')
plt.plot((AHist[3][1:]-A_dx0[1:])/(.1), label = '50')

plt.plot(np.zeros(100), 'k')

plt.ylim([-4,5])
plt.legend()
plt.show()



'''

mNrm = ss.state_now['mNrm']  
mLvl = ss.state_now['mNrm']*ss.state_now['pLvl']
plt.hist(mNrm, bins=np.linspace(0,5,num=1000))
plt.show()

plt.hist(mLvl, bins=np.linspace(0,1.2,num=1000))
plt.show()

'''








