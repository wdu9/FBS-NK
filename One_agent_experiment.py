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



def combine_indep_dstns(*distributions, seed=0):
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




class Test_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                 
                                                   "B",
                                                   "cList",
                                                   "s",
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "Ghost",
                                                   "G",
                                                   "jacW"
                                                  
                                                  ]
    

    
    
    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 0, **kwds)
        
        self.cList = []
        solver = FBSNK_solver
        self.solve_one_period = make_one_period_oo_solver(solver)
        
    '''
        
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ))/ (self.wage*self.tax_rate)  #calculate SS labor supply from Budget Constraint

        
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
     '''  
    

    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        
        
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
        PermShkDstnW     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnW = [combine_indep_dstns(PermShkDstnW,TranShkDstnW)]
        self.TranShkDstnW = [TranShkDstnW]
        self.PermShkDstnW = [PermShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
        
    
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
        # Get and store states for newly born agents
        N = np.sum(which_agents)  # Number of new consumers to make
        
        aNrmDstn =Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1),
        ).approx(N)
        
        self.state_now['aNrm'][which_agents] = aNrmDstn.draw(N,exact_match=True)
        
        '''
        self.state_now['aNrm'][which_agents] = Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1),
        ).draw(N)
        '''
        
        # why is a now variable set here? Because it's an aggregate.
        pLvlInitMeanNow = self.pLvlInitMean + np.log(
            self.state_now['PlvlAgg']
        )  # Account for newer cohorts having higher permanent income
        
        
        pLvlDstn=Lognormal(
            pLvlInitMeanNow,
            self.pLvlInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).approx(N)
        self.state_now['pLvl'][which_agents]=pLvlDstn.draw(N,exact_match=True)
        
        '''
        self.state_now['pLvl'][which_agents] = Lognormal(
            pLvlInitMeanNow,
            self.pLvlInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)
        
        '''
        self.t_age[which_agents] = 0  # How many periods since each agent was born
        self.t_cycle[
            which_agents
        ] = 0  # Which period of the cycle each agent is currently in
        return None
    
    def sim_death(self):
        """
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.
        Parameters
        ----------
        None
        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Determine who dies
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        DiePrb = DiePrb_by_t_cycle[
            self.t_cycle - 1
        ]  # Time has already advanced, so look back one
        
        DeathDstn=Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).approx(N=self.AgentCount)
        DeathShks = DeathDstn.draw(N=self.AgentCount,exact_match=True)
        
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents
    
    
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
                IncShks = IncShkDstnNow.draw(N, exact_match=True) 

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
            EventDraws = IncShkDstnNow.draw_perf(N)
            #EventDraws = IncShkDstnNow.draw_events(N)
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
    
        
        if self.jac==True or self.Ghost==True:
            RfreeNow = self.Rfree[self.t_sim]* np.ones(self.AgentCount)
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
        
        
        if self.jac == True or self.Ghost == True or self.jacW == True:
            if self.t_sim == 0:
                
                    mNrmNow = ss.state_now['mNrm']
                    pLvlNow = ss.state_now['pLvl']

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None




    
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                            # Coefficient of relative risk aversion
    "Rfree": 1.048**.25,                   # Interest factor on assets
    "DiscFac": 0.9671,                     # Intertemporal discount factor
    "LivPrb" : [.995],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.09535573122529638,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.16563445378151262,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 5,                       # Maximum end-of-period "assets above minimum" value
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
    "T_sim" : 100,                          # Number of periods to simulate
    "aNrmInitMean" : np.log(1.5)-(.5**2)/2,                 # Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     "s"          : 7,
     "dx"         : 0,                     #Deviation from steady state
     "jac"        : False,
     "Ghost"      : False,
     "jacW"       : False,
     
     
    #New Economy Parameters
     "SSWmu " : 1.1 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.012,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0 ,                              # Net Bond Supply
     "G" : 0.19
}


###############################################################################


G=.19
t=0.16563445378151262
Inc = 0.09535573122529638
mho=.05

w = (1/1.012)
N = (Inc*mho + G) / (w*t) 
r = (1.048)**.25 - 1 
q = ((1-w)*N)/r

print(N)
print(q)

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
ss.jac= False 

ss.cycles = 0
ss.dx = 0
ss.T_sim= 1700




########################################################################
target = q
go=True
tolerance=.00001
completed_loops=0

while go:
    

    ss.solve()
    ss.initialize_sim()
    ss.simulate()

    


    AggA = np.mean(ss.state_now['aLvl'])
    AggC = np.mean((ss.state_now['mNrm'] - ss.state_now['aNrm'])  * ss.state_now['pLvl'])

    
    
    if AggA - target > 0 :
        
        DiscFac1 = ss.DiscFac - .00001
        ss.DiscFac = (ss.DiscFac + DiscFac1)/2
        
    elif AggA - target < 0: 
        DiscFac2 = ss.DiscFac + .00001
        ss.DiscFac =  (ss.DiscFac + DiscFac2)/2
        
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
    
    go = distance >= tolerance and completed_loops < 1     
print("Done Computing Steady State")







###############################################################################

class Test_agent2(Test_agent):
    
    
    
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
        
        self.solution_terminal.vFunc = deepcopy(ss.solution[0].vFunc)
        self.solution_terminal.vPfunc = deepcopy(ss.solution[0].vPfunc)
        self.solution_terminal.vPPfunc =  deepcopy(ss.solution[0].vPPfunc)
     
        
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
                IncShks = IncShkDstnNow.draw(N, exact_match=True) 

                PermShkNow[these] = (
                    IncShks[0, :] * PermGroFacNow
                )  # permanent "shock" includes expected growth
                TranShkNow[these] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstnNow = ss.IncShkDstn[0]  # set current income distribution
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor
           

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstnNow.draw_perf(N)
            #EventDraws = IncShkDstnNow.draw_events(N)
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

    
    

##############################################################################
listC_g = []
listA_g = []
listM_g = []
listmNrm_g=[]
listaNrm_g=[]

params = deepcopy(IdiosyncDict)

params['T_cycle']= 200
params['LivPrb']= params['T_cycle']*[ss.LivPrb[0]]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[(0.01*4/11)**0.5]
params['TranShkStd']= params['T_cycle']*[.2]
params['Rfree'] = params['T_cycle']*[ss.Rfree]

ss_dx = Test_agent2(**params )
ss_dx.pseudo_terminal = False
ss_dx.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']

#ss_dx.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)


ss_dx.jac = False
ss_dx.Ghost = True
ss_dx.jacW = False
ss_dx.T_sim = params['T_cycle']
ss.cList=[]
ss_dx.dx = 0
ss_dx.cycles= 1


ss_dx.IncShkDstn = params['T_cycle']*ss_dx.IncShkDstn
ss_dx.del_from_time_inv('Rfree')
ss_dx.add_to_time_vary('Rfree')


ss_dx.solve()
ss_dx.initialize_sim()
ss_dx.simulate()


for j in range(ss_dx.T_sim):

    aNrmg = np.mean(ss_dx.history['aNrm'][j,:])
    mNrmg = np.mean(ss_dx.history['mNrm'][j,:])
    Mg = np.mean(ss_dx.history['mNrm'][j,:]*ss_dx.history['pLvl'][j,:])
    Cg = np.mean(ss_dx.history['cNrm'][j,:]*ss_dx.history['pLvl'][j,:])
    Ag = np.mean(ss_dx.history['aLvl'][j,:])

    listaNrm_g.append(aNrmg)
    listmNrm_g.append(mNrmg)
    listM_g.append(Mg)
    listA_g.append(Ag)
    listC_g.append(Cg)
    
    
aNrm_dx0 = np.array(listaNrm_g)
mNrm_dx0 = np.array(listmNrm_g)
M_dx0 = np.array(listM_g)
A_dx0 = np.array(listA_g)
C_dx0 = np.array(listC_g)

plt.plot(C_dx0, label="steady state")
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
example.dx = .1
example.update_income_process()
example.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']


if example.jac == True:
    
    example.del_from_time_inv('Rfree')
    example.add_to_time_vary('Rfree')
    example.IncShkDstn = params['T_cycle']*example.IncShkDstn

if example.jacW == True:
    example.Rfree = ss.Rfree
    




Test_set=[0,20,50,75]

Mega_list =[]
CHist = []
AHist =[]
MHist =[]
mNrmHist=[]
aNrmHist=[]

for i in Test_set:
    
        listM = []
        listC = []
        listA = []
        listmNrm =[]
        listaNrm =[]
        
        
       
        if example.jac==True:
            example.Rfree = i *[ss.Rfree] + [ss.Rfree + example.dx] + (params['T_cycle']  - i - 1)*[ss.Rfree]
        
        if example.jacW == True:
            example.IncShkDstn = i *ss.IncShkDstn + example.IncShkDstnW + (params['T_cycle'] - i - 1)* ss.IncShkDstn
        

            
        example.solve()
        example.initialize_sim()
        example.simulate()
        

        for j in range(example.T_sim):
            
            aNrm = np.mean(example.history['aNrm'][j,:])
            mNrm = np.mean(example.history['mNrm'][j,:])
            m = np.mean(example.history['mNrm'][j,:]*example.history['pLvl'][j,:])
            a = np.mean(example.history['aLvl'][j,:])
            c = np.mean(example.history['cNrm'][j,:]*example.history['pLvl'][j,:])
        
            listaNrm.append(aNrm)
            listmNrm.append(mNrm)
            listM.append(m)
            listC.append(c)
            listA.append(a)
        
        aNrmHist.append(np.array(listaNrm))
        mNrmHist.append(np.array(listmNrm))
        MHist.append(np.array(listA))
        AHist.append(np.array(listA))
        CHist.append(np.array(listC))
        #Mega_list.append(np.array(listC)- C_dx0)  # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s

        print(i)
        
        
plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[1], label = '20')
plt.plot(CHist[2], label = '50')
plt.plot(CHist[0], label = '0')
plt.legend()
plt.show()
 

plt.plot((CHist[3]-C_dx0)/(example.dx), label = '75')
plt.plot((CHist[1]-C_dx0)/(example.dx), label = '20')
plt.plot((CHist[2]-C_dx0)/(example.dx), label = '50')
plt.plot((CHist[0]-C_dx0)/(example.dx), label = '0')
plt.plot(np.zeros(params['T_cycle']), 'k')
plt.legend()
plt.show()


# 
# 
# 

plt.plot(M_dx0 , label = 'Steady State')
plt.plot(MHist[1], label = '5')
plt.plot(MHist[2], label = '20')
plt.plot(MHist[0], label = '1')
plt.ylim([0,4])
plt.legend()
plt.show()

# 
plt.plot((MHist[1]-M_dx0)/(.1), label = '5')
plt.plot((MHist[2]-M_dx0)/(.1), label = '20')
plt.plot((MHist[0]-M_dx0)/(.1), label = '1')
plt.plot(np.zeros(params['T_cycle']), 'k')
plt.ylim([-1,1])
plt.legend()
plt.show()
# 




plt.plot(A_dx0 , label = 'Steady State')
plt.plot(AHist[1], label = '20')
plt.plot(AHist[3], label = '75')
plt.plot(AHist[2], label = '50')
plt.plot(AHist[0], label = '0')
plt.legend()
plt.show()



plt.plot((AHist[0]-A_dx0)/(example.dx), label = '0')
plt.plot((AHist[1]-A_dx0)/(example.dx), label = '20')
plt.plot((AHist[2]-A_dx0)/(example.dx), label = '50')
plt.plot((AHist[3]-A_dx0)/(example.dx), label = '75')
plt.plot(np.zeros(params['T_cycle']), 'k')
plt.legend()

plt.ylim([-.1,.1])
plt.legend()
plt.show()





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


plt.plot(np.zeros(example.T_cycle), label = 'Steady State')
plt.plot((aNrmHist[1]-aNrm_dx0)/example.dx, label = '20')
plt.plot((aNrmHist[2]-aNrm_dx0)/example.dx, label = '50')
plt.plot((aNrmHist[3]-aNrm_dx0)/example.dx, label = '75')
plt.plot((aNrmHist[0]-aNrm_dx0)/example.dx, label = '0')
plt.ylim([-.1,.1])
plt.legend()
plt.show()


'''
Shks=[]
for i in range(100):
    Shks.append(ss.IncShkDstn[0].draw(1000))
    
mean=[]
meanp=[]
for i in range(100):
    mean.append(np.mean(Shks[i][1]))
    meanp.append(np.mean(Shks[i][0]))
'''


