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


import HARK.ConsumptionSaving.ConsIndShockModel2 as ConsIndShockModel
from HARK.ConsumptionSaving.ConsIndShockModel2 import (
    ConsIndShockSolver,
    IndShockConsumerType,
    PerfForesightConsumerType,
)

from HARK import Market, make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform, calc_expectation, Distribution

from HARK.core import solve_one_cycle







class DiscreteDistribution(Distribution):
    """
    A representation of a discrete probability distribution.
    Parameters
    ----------
    pmf : np.array
        An array of floats representing a probability mass function.
    X : np.array or [np.array]
        Discrete point values for each probability mass.
        May be multivariate (list of arrays).
    seed : int
        Seed for random number generator.
    """

    pmf = None
    X = None

    def __init__(self, pmf, X, seed=0):
        self.pmf = pmf
        self.X = X
        # Set up the RNG
        super().__init__(seed)

        # Very quick and incomplete parameter check:
        # TODO: Check that pmf and X arrays have same length.

    def dim(self):
        if isinstance(self.X, list):
            return len(self.X)
        else:
            return 1

    def draw_events(self, n):
        """
        Draws N 'events' from the distribution PMF.
        These events are indices into X.
        """
        # Generate a cumulative distribution
        base_draws = self.RNG.uniform(size=n)
        cum_dist = np.cumsum(self.pmf)

        # Convert the basic uniform draws into discrete draws
        indices = cum_dist.searchsorted(base_draws)

        return indices
    
    def draw_perf(self, N, X=None, exact_match=True):
        
        """
        Simulates N draws from a discrete distribution with probabilities P and outcomes X.
        Parameters
        ----------
        N : int
            Number of draws to simulate.
        X : None, int, or np.array
            If None, then use this distribution's X for point values.
            If an int, then the index of X for the point values.
            If an np.array, use the array for the point values.
        exact_match : boolean
            Whether the draws should "exactly" match the discrete distribution (as
            closely as possible given finite draws).  When True, returned draws are
            a random permutation of the N-length list that best fits the discrete
            distribution.  When False (default), each draw is independent from the
            others and the result could deviate from the input.
        Returns
        -------
        draws : np.array
            An array of draws from the discrete distribution; each element is a value in X.
        """
        if X is None:
            X = self.X
            J = self.dim()
        elif isinstance(X, int):
            X = self.X[X]
            J = 1
        else:
            X = X
            J = 1

        if exact_match:
            events = np.arange(self.pmf.size)  # just a list of integers
            cutoffs = np.round(np.cumsum(self.pmf) * N).astype(
                int
            )  # cutoff points between discrete outcomes
            top = 0

            # Make a list of event indices that closely matches the discrete distribution
            event_list = []
            for j in range(events.size):
                bot = top
                top = cutoffs[j]
                event_list += (top - bot) * [events[j]]
                
            indices = self.RNG.permutation(event_list)

            # Randomly permute the event indices
            
        return indices 

    def draw(self, N, X=None, exact_match=True):
        """
        Simulates N draws from a discrete distribution with probabilities P and outcomes X.
        Parameters
        ----------
        N : int
            Number of draws to simulate.
        X : None, int, or np.array
            If None, then use this distribution's X for point values.
            If an int, then the index of X for the point values.
            If an np.array, use the array for the point values.
        exact_match : boolean
            Whether the draws should "exactly" match the discrete distribution (as
            closely as possible given finite draws).  When True, returned draws are
            a random permutation of the N-length list that best fits the discrete
            distribution.  When False (default), each draw is independent from the
            others and the result could deviate from the input.
        Returns
        -------
        draws : np.array
            An array of draws from the discrete distribution; each element is a value in X.
        """
        if X is None:
            X = self.X
            J = self.dim()
        elif isinstance(X, int):
            X = self.X[X]
            J = 1
        else:
            X = X
            J = 1

        if exact_match:
            events = np.arange(self.pmf.size)  # just a list of integers
            cutoffs = np.round(np.cumsum(self.pmf) * N).astype(
                int
            )  # cutoff points between discrete outcomes
            top = 0

            # Make a list of event indices that closely matches the discrete distribution
            event_list = []
            for j in range(events.size):
                bot = top
                top = cutoffs[j]
                event_list += (top - bot) * [events[j]]

            # Randomly permute the event indices
            indices = self.RNG.permutation(event_list)

        # Draw event indices randomly from the discrete distribution
        else:
            indices = self.draw_events(N)

        # Create and fill in the output array of draws based on the output of event indices
        if J > 1:
            draws = np.zeros((J, N))
            for j in range(J):
                draws[j, :] = X[j][indices]
        else:
            draws = np.asarray(X)[indices]

        return draws



def combine_indep_dstns2(*distributions, seed=0):
    """
    Given n lists (or tuples) whose elements represent n independent, discrete
    probability spaces (probabilities and values), construct a joint pmf over
    all combinations of these independent points.  Can take multivariate discrete
    distributions as inputs.
    Parameters
    ----------
    distributions : [np.array]
        Arbitrary number of distributions (pmfs).  Each pmf is a list or tuple.
        For each pmf, the first vector is probabilities and all subsequent vectors
        are values.  For each pmf, this should be true:
        len(X_pmf[0]) == len(X_pmf[j]) for j in range(1,len(distributions))
    Returns
    -------
    A DiscreteDistribution, consisting of:
    P_out: np.array
        Probability associated with each point in X_out.
    X_out: np.array (as many as in *distributions)
        Discrete points for the joint discrete probability mass function.
    """
    # Get information on the distributions
    dist_lengths = ()
    dist_dims = ()
    for dist in distributions:
        dist_lengths += (len(dist.pmf),)
        dist_dims += (dist.dim(),)
    number_of_distributions = len(distributions)

    # Initialize lists we will use
    X_out = []
    P_temp = []

    # Now loop through the distributions, tiling and flattening as necessary.
    for dd, dist in enumerate(distributions):

        # The shape we want before we tile
        dist_newshape = (
            (1,) * dd + (len(dist.pmf),) + (1,) * (number_of_distributions - dd)
        )

        # The tiling we want to do
        dist_tiles = dist_lengths[:dd] + (1,) + dist_lengths[dd + 1 :]

        # Now we are ready to tile.
        # We don't use the np.meshgrid commands, because they do not
        # easily support non-symmetric grids.

        # First deal with probabilities
        Pmesh = np.tile(dist.pmf.reshape(dist_newshape), dist_tiles)  # Tiling
        flatP = Pmesh.ravel()  # Flatten the tiled arrays
        P_temp += [
            flatP,
        ]  # Add the flattened arrays to the output lists

        # Then loop through each value variable
        for n in range(dist_dims[dd]):
            if dist.dim() > 1:
                Xmesh = np.tile(dist.X[n].reshape(dist_newshape), dist_tiles)
            else:
                Xmesh = np.tile(dist.X.reshape(dist_newshape), dist_tiles)
            flatX = Xmesh.ravel()
            X_out += [
                flatX,
            ]

    # We're done getting the flattened X_out arrays we wanted.
    # However, we have a bunch of flattened P_temp arrays, and just want one
    # probability array. So get the probability array, P_out, here.
    P_out = np.prod(np.array(P_temp), axis=0)

    assert np.isclose(np.sum(P_out), 1), "Probabilities do not sum to 1!"
    return DiscreteDistribution(P_out, X_out, seed=seed)








################################################################################





class Test_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                 
                                                   "B",
                                                  
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "Ghost",
                                                   "G",
                                                   "jacW",
                                                   "jacN"
                                                   
                                                   
                                                  
                                                  ]
    

    
    
    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 0, **kwds)

    
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G + (1 - (1/(self.Rfree) ) ) * self.B) / (self.wage*self.tax_rate)
        
        
        TranShkDstn     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstn.pmf  = np.insert(TranShkDstn.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstn.X  = np.insert(TranShkDstn.X*(((1.0-self.tax_rate)*self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstn     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstn = [combine_indep_dstns(PermShkDstn,TranShkDstn)]
        self.TranShkDstn = [TranShkDstn]
        self.PermShkDstn = [PermShkDstn]
        self.add_to_time_vary('IncShkDstn')
        
  
        TranShkDstnW     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnW.pmf  = np.insert(TranShkDstnW.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnW.X  = np.insert(TranShkDstnW.X*(((1.0-self.tax_rate)*self.N*(self.wage + self.dx))/(1-self.UnempPrb)),0,self.IncUnemp)
        self.IncShkDstnW = [combine_indep_dstns(PermShkDstn,TranShkDstnW)]
        self.TranShkDstnW = [TranShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
        
        TranShkDstnN     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnN.pmf  = np.insert(TranShkDstnN.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnN.X  = np.insert(TranShkDstnN.X*(((1.0-self.tax_rate)*(self.N + self.dx)*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        self.IncShkDstnN = [combine_indep_dstns(PermShkDstn,TranShkDstnN)]
        self.TranShkDstnN = [TranShkDstnN]
        self.add_to_time_vary('IncShkDstnN')
        
        TranShkDstnP     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnP.pmf  = np.insert(TranShkDstnP.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnP.X  = np.insert(TranShkDstnP.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnP     = MeanOneLogNormal(self.PermShkStd[0] + self.dx ,123).approx(self.PermShkCount)
        self.IncShkDstnP= [combine_indep_dstns2(PermShkDstnP,TranShkDstnP)]
        self.TranShkDstnP = [TranShkDstnP]
        self.PermShkDstnP = [PermShkDstnP]
        self.add_to_time_vary('IncShkDstnP')
     
        

    



    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                             # Coefficient of relative risk aversion
    "Rfree": 1.05**.25,                  # Interest factor on assets
    "DiscFac": 0.9811,                    # Intertemporal discount factor
    "LivPrb" : [.99375],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [.06],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],                   # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.2,      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.2,      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 15,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other parameters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type

    # Parameters only used in simulation
    "AgentCount" : 150000,                 # Number of agents of this type
    "T_sim" : 100,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(1.7)-(.5**2)/2,# Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0,                    # Mean of log initial permanent income
    "pLvlInitStd"  : 0,                    # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     "dx"         : 0,                     #Deviation from steady state
     "jac"        : False,
     "Ghost"      : False,
     "jacW"       : False,
     "jacN"       : False,
     "jacPerm"    : False,

     
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.012,                     # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0.5 ,                             # Net Bond Supply
     "G" : 0.23
}


###############################################################################


#G=.5
#t =.25
#Inc = 0.1
#mho=.05

#w = (1/1.015)
#N = (Inc*mho + G) / (w*t) 
#r = (1.04)**.25 - 1 
#q = ((1-w)*N)/r

#print(N)
#print(q)


G=.23
t=0.2
Inc = 0.2
mho=.05
r = (1.05)**.25 - 1 
B=.5 #.65

w = (1/1.012)
N = (Inc*mho + G  + (1 - (1/(1+r)) ) *B) / (w*t) 
q = ((1-w)*N)/r

A = ( B/(1+r) ) + q

print(N)
print(q)
print(A)


'''
N= 1 + G

tnew = (Inc*mho + G)/(N*w)
print(tnew)

new = (N*w*tnew - G) / (mho)
print(new)

print(N)
print(N-G)
print(q)
'''


####################################################################


ss = Test_agent(**IdiosyncDict )
ss.cycles = 0
ss.dx = 0
ss.T_sim= 1200
ss.aNrmInitMean = np.log(A)-(.5**2)/2

########################################################################

target = A
go=True
tolerance=.01
completed_loops=0

while go:
    

    ss.solve()
    ss.initialize_sim()
    ss.simulate()
    

    AggA = np.mean(ss.state_now['aLvl'])
    AggC = np.mean((ss.state_now['mNrm'] - ss.state_now['aNrm'])  * ss.state_now['pLvl'])

    dif = AggA - target

    if dif > 0 :
        ss.DiscFac = ss.DiscFac - dif/200
    elif dif < 0: 
        ss.DiscFac = ss.DiscFac - dif/200
    else:
        break

    
    print('Assets')
    print(AggA)
    print('consumption')
    print(AggC)
    print('center')
    print(ss.DiscFac)
    
    distance = abs(AggA - target) 
    
    completed_loops += 1
    
    print('Completed loops')
    print(completed_loops)
    
    go = distance >= tolerance and completed_loops < 10
     
print("Done Computing Steady State")



###############################################################################

class Test_agent2(Test_agent):
    
    
     def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        
        self.N = ss.N
        
        
        TranShkDstn     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstn.pmf  = np.insert(TranShkDstn.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstn.X  = np.insert(TranShkDstn.X*(((1.0-self.tax_rate)*self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstn     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstn = [combine_indep_dstns(PermShkDstn,TranShkDstn)]
        self.TranShkDstn = [TranShkDstn]
        self.PermShkDstn = [PermShkDstn]
        self.add_to_time_vary('IncShkDstn')
        
  
        TranShkDstnW     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnW.pmf  = np.insert(TranShkDstnW.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnW.X  = np.insert(TranShkDstnW.X*(((1.0-self.tax_rate)*self.N*(self.wage + self.dx))/(1-self.UnempPrb)),0,self.IncUnemp)
        self.IncShkDstnW = [combine_indep_dstns(PermShkDstn,TranShkDstnW)]
        self.TranShkDstnW = [TranShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
        
        TranShkDstnN     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnN.pmf  = np.insert(TranShkDstnN.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnN.X  = np.insert(TranShkDstnN.X*(((1.0-self.tax_rate)*(self.N + self.dx)*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        self.IncShkDstnN = [combine_indep_dstns(PermShkDstn,TranShkDstnN)]
        self.TranShkDstnN = [TranShkDstnN]
        self.add_to_time_vary('IncShkDstnN')
        
        TranShkDstnP     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnP.pmf  = np.insert(TranShkDstnP.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnP.X  = np.insert(TranShkDstnP.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnP     = MeanOneLogNormal(self.PermShkStd[0] + self.dx ,123).approx(self.PermShkCount)
        self.IncShkDstnP= [combine_indep_dstns2(PermShkDstnP,TranShkDstnP)]
        self.TranShkDstnP = [TranShkDstnP]
        self.PermShkDstnP = [PermShkDstnP]
        self.add_to_time_vary('IncShkDstnP')
    
    
    
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
        self.solution_terminal.cFunc = deepcopy(ss.solution[0].cFunc)
        self.solution_terminal.vFunc = deepcopy(ss.solution[0].vFunc)
        self.solution_terminal.vPfunc = deepcopy(ss.solution[0].vPfunc)
        self.solution_terminal.vPPfunc =  deepcopy(ss.solution[0].vPPfunc)
     



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
            
                mNrmNow = ss.state_now['mNrm']
                pLvlNow = ss.state_now['pLvl']

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None

    
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
    
        
        if self.jac==True or self.Ghost==True:
            RfreeNow = self.Rfree[self.t_sim]* np.ones(self.AgentCount)
        else:
            RfreeNow = ss.Rfree * np.ones(self.AgentCount)
            
        return RfreeNow
        
    
    

    

##############################################################################
listC_g = []
listA_g = []
listM_g = []
listmNrm_g=[]
listaNrm_g=[]
listMU_g =[]

params = deepcopy(IdiosyncDict)

params['T_cycle']= 200
params['LivPrb']= params['T_cycle']*[ss.LivPrb[0]]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[(0.01*4/11)**0.5]
params['TranShkStd']= params['T_cycle']*[.2]
params['Rfree'] = params['T_cycle']*[ss.Rfree]

ss_dx = Test_agent2(**params )
ss_dx.pseudo_terminal = False
ss_dx.PerfMITShk = True
ss_dx.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl','TranShk']

ss_dx.Ghost = True
ss_dx.T_sim = params['T_cycle']
ss.cList=[]
ss_dx.dx = 0
ss_dx.cycles= 1


ss_dx.IncShkDstn = params['T_cycle']*ss_dx.IncShkDstn
ss_dx.del_from_time_inv('Rfree')
ss_dx.add_to_time_vary('Rfree')

ss_dx.solve()

#ss_dx.LivPrb = params['T_cycle']*[1]

ss_dx.initialize_sim()
ss_dx.simulate()



norm = ((1-ss.UnempPrb)/((ss.wage) * ss.N * (1 - ss.tax_rate)))


for j in range(ss_dx.T_sim):

    aNrmg = np.mean(ss_dx.history['aNrm'][j,:])
    mNrmg = np.mean(ss_dx.history['mNrm'][j,:])
    Mg = np.mean(ss_dx.history['mNrm'][j,:]*ss_dx.history['pLvl'][j,:])
    Cg = np.mean(ss_dx.history['cNrm'][j,:]*ss_dx.history['pLvl'][j,:])
    Ag = np.mean(ss_dx.history['aLvl'][j,:])
    emp = ss_dx.history['TranShk'][j,:] != ss.IncUnemp
    ss_dx.history['TranShk'][j,:][emp] = norm*ss_dx.history['TranShk'][j,:][emp]
    MUg = np.mean(ss_dx.history['TranShk'][j,:][emp] * ss_dx.history['pLvl'][j,:][emp]* (ss_dx.history['cNrm'][j,:][emp]*ss_dx.history['pLvl'][j,:][emp])**(- ss.CRRA))

    listaNrm_g.append(aNrmg)
    listmNrm_g.append(mNrmg)
    listM_g.append(Mg)
    listA_g.append(Ag)
    listC_g.append(Cg)
    listMU_g.append(MUg)
    

aNrm_dx0 = np.array(listaNrm_g)
mNrm_dx0 = np.array(listmNrm_g)
M_dx0 = np.array(listM_g)
A_dx0 = np.array(listA_g)
C_dx0 = np.array(listC_g)
MU_dx0 = np.array(listMU_g)


plt.plot(C_dx0, label="steady state")
plt.plot(MU_dx0 , label = 'Marginal Utility steady state' )
plt.legend()
plt.show()

print('done with Ghost Run')

##############################################################################

example = Test_agent2(**params )
example.pseudo_terminal=False 

#example.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)
#example.solution_terminal = deepcopy(ss.solution[0])

example.T_sim = params['T_cycle']
example.cycles = 1
example.jac = True
example.jacW = False
example.jacN = False
example.jacPerm = False
example.PerfMITShk = True
example.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl','TranShk']


if example.jac == True:
    example.dx = .01
    example.del_from_time_inv('Rfree')
    example.add_to_time_vary('Rfree')
    example.IncShkDstn = params['T_cycle']*example.IncShkDstn

if example.jacW == True or example.jacN==True or example.jacPerm ==True:
    example.dx = .2
    example.Rfree = ss.Rfree
    example.update_income_process()
    




Test_set=[0,20,50]

Mega_list =[]
CHist = []
AHist =[]
MHist =[]
mNrmHist=[]
aNrmHist=[]
MUHist=[]

for i in Test_set:
    
        listM = []
        listC = []
        listA = []
        listmNrm =[]
        listaNrm =[]
        listMU =[]
        
       
        if example.jac==True:
            example.Rfree = i *[ss.Rfree] + [ss.Rfree + example.dx] + (params['T_cycle']  - i - 1)*[ss.Rfree]
        
        if example.jacW == True:
            example.IncShkDstn = i *ss.IncShkDstn + example.IncShkDstnW + (params['T_cycle'] - i - 1)* ss.IncShkDstn
        
        if example.jacN == True:
            example.IncShkDstn = i *ss.IncShkDstn + example.IncShkDstnN + (params['T_cycle'] - i - 1)* ss.IncShkDstn
            
        if example.jacPerm == True:
            example.IncShkDstn = i *ss.IncShkDstn + example.IncShkDstnP + (params['T_cycle'] - i - 1)* ss.IncShkDstn
        

        example.solve()
        example.initialize_sim()
        example.simulate()
        
        
        
        
        for j in range(example.T_sim):
                
            if example.jacW == True:
                
                if j == i + 1 :
                    norm1 =  ((1-ss.UnempPrb)/( (ss.wage + example.dx) * ss.N * (1 - ss.tax_rate)))

                else:
                     norm1 =norm
                     
                     
            elif example.jacN == True:
                
                if j == i+ 1:
                    norm1 =  ((1-ss.UnempPrb)/( (ss.wage) * (ss.N + example.dx) * (1 - ss.tax_rate) ) )
                else:
                  norm1 =norm
                    
            else:
                norm1 =norm
            
            
            emp = example.history['TranShk'][j] != ss.IncUnemp
            example.history['TranShk'][j][emp] = norm1 * example.history['TranShk'][j][emp]

            
        

        for j in range(example.T_sim):
            
            aNrm = np.mean(example.history['aNrm'][j,:])
            mNrm = np.mean(example.history['mNrm'][j,:])
            m = np.mean(example.history['mNrm'][j,:]*example.history['pLvl'][j,:])
            a = np.mean(example.history['aLvl'][j,:])
            c = np.mean(example.history['cNrm'][j,:]*example.history['pLvl'][j,:])
            
            emp = example.history['TranShk'][j,:] != ss.IncUnemp
            MU = np.mean(example.history['TranShk'][j,:][emp] * example.history['pLvl'][j,:][emp]* (example.history['cNrm'][j,:][emp]*example.history['pLvl'][j,:][emp])**(- ss.CRRA))
        
            listaNrm.append(aNrm)
            listmNrm.append(mNrm)
            listM.append(m)
            listC.append(c)
            listA.append(a)
            listMU.append(MU)
            
        
        aNrmHist.append(np.array(listaNrm))
        mNrmHist.append(np.array(listmNrm))
        MHist.append(np.array(listM))
        AHist.append(np.array(listA))
        CHist.append(np.array(listC))
        MUHist.append(np.array(listMU))
        #Mega_list.append(np.array(listC)- C_dx0)  # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s

        print(i)
        
        
        
    
plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[1], label = '20')
plt.plot(CHist[2], label = '50')
plt.plot(CHist[0], label = '0')
plt.title("Aggregate Consumption (wage)")
plt.ylabel("Aggregate Consumption")
plt.xlabel("Period")
#plt.savefig("AggregateConsumption.jpg", dpi=400)
plt.legend()
plt.show()


plt.plot((CHist[1]-C_dx0)/(example.dx), label = '20')
plt.plot((CHist[2]-C_dx0)/(example.dx), label = '50')
plt.plot((CHist[0]-C_dx0)/(example.dx), label = '0')
plt.ylabel("dC / dw")
plt.xlabel("Period")
plt.title("Consumption Jacobians (wage)")
plt.plot(np.zeros(params['T_cycle']), 'k')
#plt.savefig("ConsumptionJacobian_wage.jpg", dpi=400)
plt.legend()
plt.show()


plt.plot((MUHist[1]-MU_dx0)/(example.dx), label = '20')
plt.plot((MUHist[2]-MU_dx0)/(example.dx), label = '50')
plt.plot((MUHist[0]-MU_dx0)/(example.dx), label = '0')
plt.plot(np.zeros(params['T_cycle']), 'k')
#plt.savefig("ConsumptionJacobian_wage.jpg", dpi=400)
plt.legend()
plt.show()




plt.plot(M_dx0 , label = 'Steady State')
plt.plot(MHist[1], label = '5')
plt.plot(MHist[2], label = '20')
plt.plot(MHist[0], label = '1')
plt.ylim([0,4])
plt.legend()
plt.show()

# 
plt.plot((MHist[1]-M_dx0)/(example.dx), label = '5')
plt.plot((MHist[2]-M_dx0)/(example.dx), label = '20')
plt.plot((MHist[0]-M_dx0)/(example.dx), label = '1')
plt.plot(np.zeros(params['T_cycle']), 'k')
plt.ylim([-1,1])
plt.legend()
plt.show()
# 




plt.plot(A_dx0 , label = 'Steady State')
plt.plot(AHist[1], label = '20')
plt.plot(AHist[2], label = '50')
plt.plot(AHist[0], label = '0')
plt.xlabel("Period")
plt.ylabel("Aggregate Assets")
plt.title("Aggregate Assets (wage)")
#plt.savefig("Aggregate Assets_wage.jpg", dpi=400)

plt.legend()
plt.show()



plt.plot((AHist[0]-A_dx0)/(example.dx), label = '0')
plt.plot((AHist[1]-A_dx0)/(example.dx), label = '20')
plt.plot((AHist[2]-A_dx0)/(example.dx), label = '50')
plt.ylabel("dA / dw")
plt.xlabel("Period")
plt.title("Asset Jacobians (wage)")
#plt.savefig("AssetJacobian_wage.jpg", dpi=400)

plt.plot(np.zeros(params['T_cycle']), 'k')
plt.legend()
plt.show()




'''
plt.plot(mNrm_dx0 , label = 'Steady State')
plt.plot(mNrmHist[1], label = '20')
plt.plot(mNrmHist[2], label = '50')
plt.plot(mNrmHist[3], label = '75')
plt.plot(mNrmHist[0], label = '0')
plt.ylim([2.2,2.6])
plt.legend()
plt.show()


plt.plot(aNrm_dx0 , label = 'Steady State')
plt.plot(aNrmHist[1], label = '20')
plt.plot(aNrmHist[2], label = '50')
plt.plot(aNrmHist[3], label = '75')
plt.plot(aNrmHist[0], label = '0')
plt.ylim([1,1.5])
plt.legend()
plt.show()


plt.plot(np.zeros(example.T_cycle),'k')
plt.plot((aNrmHist[1]-aNrm_dx0)/example.dx, label = '20')
plt.plot((aNrmHist[2]-aNrm_dx0)/example.dx, label = '50')
plt.plot((aNrmHist[3]-aNrm_dx0)/example.dx, label = '75')
plt.plot((aNrmHist[0]-aNrm_dx0)/example.dx, label = '0')
plt.ylim([-.1,.1])
plt.legend()
plt.show()


Shks=[]
for i in range(100):
    Shks.append(ss.IncShkDstn[0].draw(1000))
    
mean=[]
meanp=[]
for i in range(100):
    mean.append(np.mean(Shks[i][1]))
    meanp.append(np.mean(Shks[i][0]))


t_cycle_0=[]
t_age_0=[]
for i in range(50000):
    if example.t_cycle[i] == 0:
        t_cycle_0.append(example.t_cycle[i])
    if example.t_age[i] == 1:
        t_age_0.append(example.t_age[i])
        
print(len(t_cycle_0))
print(len(t_age_0))

'''