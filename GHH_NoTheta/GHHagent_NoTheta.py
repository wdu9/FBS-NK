# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 02:29:04 2021

@author: wdu
"""

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



from GHH_Utility_NoTheta import ( GHHutility,         
 GHHutilityP,
 GHHutilityPP,
 GHHutilityP_inv,
GHHutility_invP,
GHHutility_inv,
 GHHutilityP_invP,
  ValueFuncGHH,
 MargValueFuncGHH,
 MargMargValueFuncGHH
 
 )

from JAC_Utility import DiscreteDistribution2, combine_indep_dstns2


#---------------------------------------------------------------------------------



utility = GHHutility
utilityP = GHHutilityP
utilityPP = GHHutilityPP
utilityP_inv = GHHutilityP_inv
utility_invP = GHHutility_invP
utility_inv = GHHutility_inv
utilityP_invP = GHHutilityP_invP




#-----------------------------------------------------------------------------


##############################################################################






class FBSNK_Solver(ConsIndShockSolver):
           
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
                
                N,
                v,
                varphi,
                

                ):
                 

        
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
        
        self.N = N
        self.v = v
        self.varphi = varphi
    
    
    
    
    
    
    
    def def_utility_funcs(self):
        """
        Defines CRRA utility function for this period (and its derivatives),
        saving them as attributes of self for other methods to use.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        
        self.u = lambda c: utility(c, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA ) # utility function
        self.uP = lambda c: utilityP(c, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA)   # marginal utility function
        self.uPP = lambda c: utilityPP(c, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA ) # marginal marginal utility function
        
        
        self.uPinv = lambda u: utilityP_inv(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA )
        self.uPinvP = lambda u: utilityP_invP(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA )
        self.uinvP = lambda u: utility_invP(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA )
        if self.vFuncBool:
            self.uinv = lambda u: utility_inv(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA )
            
            
            
            
            


###############################################################################


class FBSNK_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                   "L",
                                                   "SSPmu",
                                                   "SSWmu",
                                                   
                                                   #"wage",
                                                   "N",
                                                   "v",
                                                   "varphi",
                                                   
                                                   "B",
                                                 
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "jacW",
                                                   "jacN",
                                                   "PermShkStd",
                                                   "Ghost",
                                                   
                                                    "PermShkCount",
                                                    "TranShkCount",
                                                    "TranShkStd",
                                                    "tax_rate",
                                                    "UnempPrb",
                                                    "IncUnemp",
                                                    "G",
                                                  
                                                
    
                                                    
                                                  ]
    
    

    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 0, **kwds)
        
        solver = FBSNK_Solver
        self.solve_one_period = make_one_period_oo_solver(solver)


        
    
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G + ( 1/(self.Rfree) - 1)* self.B ) / (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        
        #TranShkDstnTEST = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        #self.ThetaShk = np.insert(TranShkDstnTEST.X ,0, self.IncUnemp)


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
        
        
        '''      
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
  

        self.solution_terminal.vFunc = ValueFuncGHH(self.cFunc_terminal_, self.CRRA, n = self.N, varphi = self.varphi , v = self.v)
        self.solution_terminal.vPfunc = MargValueFuncGHH(self.cFunc_terminal_, self.CRRA,n = self.N, varphi = self.varphi , v = self.v)
        self.solution_terminal.vPPfunc = MargMargValueFuncGHH(
            self.cFunc_terminal_, self.CRRA,n = self.N, varphi = self.varphi , v = self.v)
    '''
        
    
    
    
FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.05**.25,                       # Interest factor on assets
    "DiscFac": 0.987,                     # Intertemporal discount factor
    "LivPrb" : [.99375],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05, #.08                     # Probability of unemployment while working
    "IncUnemp" :  0.05, #0.29535573122529635,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.2,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 12,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other parameters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type

    # Parameters only used in simulation
    "AgentCount" : 50000,                 # Number of agents of this type
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(.2)-(.5**2)/2, # Mean of log initial assets
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
     "Ghost"      : False, 
     
     
    #New Economy Parameters
     "SSWmu" : 1.05 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.012,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : .02,                               # Net Bond Supply
     "G" : .02 ,#.18
     "varphi": .8,
     "v": 2
     }


    
###############################################################################


G=.02
Inc = 0.06
mho=.05
t=.2
B = 0.018  # If B is negative here this means that the government is borrowing, 
# if it is positive then the government is lending 
r = (1.05)**.25 - 1 


w = (1/1.012)
N = (Inc*mho + G + (1/(1+r) - 1)*B ) / (w*t) 
q = ((1-w)*N)/r

A = (B/(1+r)) + q  # If B is positive, this means government is lending, which means asset managers are borrowing?
print(N)
print(A)


'''
N= 1 + G

tnew = (Inc*mho + G)/(N*w)
print(tnew)

new = (N*w*tnew - .18) / (mho)
print(new)

print(N)
print(N-G)
print(q)

'''


ss_agent = FBSNK_agent(**FBSDict)
ss_agent.cycles = 0
ss_agent.solve()

plot_funcs(ss_agent.solution[0].cFunc,-2,10)

print(ss_agent.N)
###############################################################################

ss_agent = FBSNK_agent(**FBSDict)
ss_agent.cycles = 0
ss_agent.dx = 0
ss_agent.T_sim = 1200
#ss_agent.track_vars = ['TranShk']

target = q


NumAgents = 50000

tolerance = .001

completed_loops=0

go = True

num_consumer_types = 5     # number of types 

center =.968  #98 

while go:
    
    discFacDispersion = 0.0069
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
    litMU1 =[]
    litMU2 =[]
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
        
        
        transhk = consumers_ss[i].shocks['TranShk'][emp]
        pLvl = consumers_ss[i].state_now['pLvl'][emp]
        cLvl = (consumers_ss[i].state_now['mNrm'] - consumers_ss[i].state_now['aNrm'])*consumers_ss[i].state_now['pLvl']
        cLvl = cLvl[emp]
        V = (ss_agent.varphi/(1+ss_agent.v)) * pLvl * ss_agent.N** (1+ss_agent.v) 
        
        litMU1.append(transhk*pLvl*(cLvl - V  )**(- ss_agent.CRRA))
        litMU2.append(pLvl*(cLvl - V  )**(- ss_agent.CRRA))

        
        print('one consumer solved and simulated')
    
    
    pLvl = np.concatenate(list_pLvl)
    aNrm = np.concatenate(list_aNrm)
    c = np.array(np.concatenate(litc))
    aLvl = np.concatenate(list_aLvl)

    AggA = np.mean(np.array(aLvl))
    AggC = np.mean(c)

    
    MU = np.array(np.concatenate(litMU1))
    MU = np.mean(MU)
    
    MU2 = np.array(np.concatenate(litMU2))
    MU2 = np.mean(MU2)
    
    print(MU/MU2)
    
    MRS = ( ss_agent.varphi * (ss_agent.N / (1 - ss_agent.IncUnemp))**(1 +ss_agent.v) ) * (MU/MU2)

    AMRS = ss_agent.SSWmu * ss_agent.wage * ( 1 - ss_agent.tax_rate )
    
    factor = AMRS/MRS 
    
    #disu_act = AMV/((ss_agent.N/(1-ss_agent.UnempPrb))**ss_agent.v)
    
    
    if AggA - target > 0 :
        
       center = center - .0001
        
    elif AggA - target < 0: 
        center = center + .0001
        
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




###############################################################################
###############################################################################

'''

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



x = np.linspace(0, 1.4, 1000, endpoint=True)

y=[]
for i in range(num_consumer_types):
    y.append(funcs[i](x))


h = np.histogram(mNrm, bins=np.linspace(0,1.4,num=1000))

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Cash on Hand')
plt.xlim(0,1.4)
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
        
        #TranShkDstnTEST = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        #self.ThetaShk = np.insert(TranShkDstnTEST.X ,0, self.IncUnemp)


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
        
        
        if self.jac == True or self.jacW == True or self.jacW == True or self.Ghost==True or self.jacN == True:
        
            if self.t_sim == 0:
                
                for i in range(num_consumer_types):
                    if  self.DiscFac == consumers_ss[i].DiscFac:

                        mNrmNow = consumers_ss[i].state_now['mNrm']
                        pLvlNow = consumers_ss[i].state_now['pLvl']
                        

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None
        


###############################################################################

params = deepcopy(FBSDict)
params['T_cycle']= 200
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

# now create types with different disc factors
for i in range(num_consumer_types):
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

        for i in range(ghosts[k].AgentCount):
            if ghosts[k].history['TranShk'][j][i] != ss_agent.IncUnemp:
        
                ghosts[k].history['TranShk'][j][i] = norm*ghosts[k].history['TranShk'][j][i]



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
        litMU_g.append(listH_MUg[n][j,:] *(listH_g[n][0][j,:]*listH_g[n][1][j,:])**(-1))
    
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

jac_agent = FBSNK_JAC(**params)

#jac_agent = FBSNK2(**params)
jac_agent.pseudo_terminal = False
jac_agent.PerfMITShk = True
jac_agent.jac = False
jac_agent.jacW =True
jac_agent.jacN = False

jac_agent.IncShkDstn = params['T_cycle']*jac_agent.IncShkDstn
jac_agent.T_sim = params['T_cycle']
jac_agent.cycles = 1
jac_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl','TranShk']

if jac_agent.jac == True:
    jac_agent.dx = .001
    jac_agent.del_from_time_inv('Rfree')
    jac_agent.add_to_time_vary('Rfree')
    jac_agent.IncShkDstn = params['T_cycle']*ss_agent.IncShkDstn

if jac_agent.jacW == True or jac_agent.jacN == True:
    jac_agent.dx = 1.2 #.8
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

testSet= [0,50,100]

Mega_list =[]
CHist = []
AHist = []
MHist = []
MUHist = []
pLvlHist =[]
#for i in range(jac_agent.T_sim +1):
    
    
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
            
            if jac_agent.jacW == True:
                consumers[k].IncShkDstn = i *ss_agent.IncShkDstn + jac_agent.IncShkDstnW + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn
                
            if jac_agent.jacN == True:
                consumers[k].IncShkDstn = i *ss_agent.IncShkDstn + jac_agent.IncShkDstnN + (params['T_cycle'] - i - 1)* ss_agent.IncShkDstn
            
            if jac_agent.jac == True:
                consumers[k].Rfree = (i)*[ss_agent.Rfree] + [ss_agent.Rfree + jac_agent.dx] + (params['T_cycle'] - i - 1)*[ss_agent.Rfree]
            

            consumers[k].solve()
            consumers[k].initialize_sim()
            consumers[k].simulate()
            
            for j in range(jac_agent.T_sim):
                
                if jac_agent.jacW == True:
                    
                    if j == consumers[k].s + 1:
                        norm1 =  ((1-ss_agent.UnempPrb)/((ss_agent.wage + jac_agent.dx) * ss_agent.N * (1 - ss_agent.tax_rate)))

                    else:
                         norm1 =norm
                         
                if jac_agent.jacN == True:
                    
                    if j == consumers[k].s + 1:
                        norm1 =  ((1-ss_agent.UnempPrb)/((ss_agent.wage) * (ss_agent.N + jac_agent.dx) * (1 - ss_agent.tax_rate)))

                    else:
                        norm1 =norm
                else:
                    norm1 =norm
                
                
                
                emp = consumers[k].history['TranShk'][j] != ss_agent.IncUnemp
                consumers[k].history['TranShk'][j][emp] = norm1*consumers[k].history['TranShk'][j][emp]

            
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
        #Mega_list.append(np.array(listC)- C_dx0)  # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s
        print(i)
###############################################################################
###############################################################################

'''
CJAC = []
AJAC = []
for i in range(201):
    CJAC.append((CHist[i]-C_dx0)/jac_agent.dx)
    AJAC.append((AHist[i]-A_dx0)/jac_agent.dx)
    
savemat('CJAC.mat', mdict={'CJAC' : CJAC})
savemat('AJAC.mat', mdict={'AJAC' : AJAC})
'''

plt.plot((MUHist[0][1:]), label = '0')
#plt.plot((CHist[4]- C_dx0)/(jac_agent.dx), label = '175')
plt.plot((MUHist[3][1:]), label = '100')
plt.plot((MUHist[1][1:]), label = '20')
plt.plot((MUHist[2][1:] ), label = '50')



plt.plot((MUHist[0][1:]- MU_dx0[1:])/(jac_agent.dx), label = '0')
#plt.plot((CHist[4]- C_dx0)/(jac_agent.dx), label = '175')
plt.plot((MUHist[3][1:]- MU_dx0[1:])/(jac_agent.dx), label = '100')
plt.plot((MUHist[1][1:]- MU_dx0[1:])/(jac_agent.dx), label = '20')
plt.plot((MUHist[2][1:] - MU_dx0[1:])/(jac_agent.dx), label = '50')




plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[1], label = '20')
#plt.plot(CHist[3], label = '100')
plt.plot(CHist[2], label = '50')
plt.title("Aggregate Consumption")
plt.ylabel("Aggregate Consumption")
plt.xlabel("Period")
plt.ylim([.9,1.2])
plt.legend()
#plt.savefig("AggregateConsumption.jpg", dpi=500)
plt.show()




plt.plot((CHist[0]- C_dx0)/(jac_agent.dx), label = '0')
#plt.plot((CHist[4]- C_dx0)/(jac_agent.dx), label = '175')
#plt.plot((CHist[3]- C_dx0)/(jac_agent.dx), label = '100')
plt.plot((CHist[1]- C_dx0)/(jac_agent.dx), label = '20')
plt.plot((CHist[2] - C_dx0)/(jac_agent.dx), label = '50')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.ylabel("dC / dw")
plt.ylim([-.01,.1])

plt.xlabel("Period")
plt.title("Consumption Jacobians")
plt.legend()
#plt.savefig("ConsumptionJacobianWage.jpg", dpi=500)
plt.show()




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
plt.plot(MHist[3] - M_dx0, label = '100')
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
plt.ylim([0,4])
plt.legend()
#plt.savefig("Aggregate Assets.jpg", dpi=500)
plt.show()




plt.plot((AHist[0]- A_dx0)/(jac_agent.dx), label = '0')
plt.plot((AHist[1]- A_dx0)/(jac_agent.dx), label = '20')
plt.plot((AHist[2] - A_dx0)/(jac_agent.dx), label = '50')
plt.plot((AHist[3]- A_dx0)/(jac_agent.dx), label = '100')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.xlabel("Period")
plt.ylabel("dA / dw")
plt.title("Asset Jacobians")
plt.legend()
#plt.savefig("Wage_AssetJacobian.jpg", dpi=500)
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


mu=.2
sigma1= .5
mean=np.log(mu)-(sigma1**2)/2

print(((np.exp(sigma1)-1)*np.exp(2*mean+sigma1**2))**.5)

print(np.exp(mean + (sigma1**2)/2))




ad = Lognormal(mu=mean, sigma=sigma1,
            seed=123,
        ).draw(100000)

plt.hist(ad,bins=np.linspace(0,1,num=10000))

plt.hist(aLvl,bins=np.linspace(0,1,num=10000))




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








