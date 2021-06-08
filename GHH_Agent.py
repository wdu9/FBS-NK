# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:12:11 2021

@author: wdu

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



from GHH_Utility import ( GHHutility,         
 GHHutilityP,
 GHHutilityPP,
 GHHutilityP_inv,
GHHutility_invP,
GHHutility_inv,
 GHHutilityP_invP,
 ValueFuncGHH,
 MargValueFuncGHH,
 MargMargValueFuncGHH,
 
 )


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
                
                ThetaShk,
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
        
        self.ThetaShk = ThetaShk
        self.N =N
        self.v =v
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
        
        self.u = lambda c: utility(c, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk , mu = self.ThetaShk[0])  # utility function
        self.uP = lambda c: utilityP(c, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk , mu = self.ThetaShk[0])  # marginal utility function
        self.uPP = lambda c: utilityPP(c, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk , mu = self.ThetaShk[0]) # marginal marginal utility function
        
        
        self.uPinv = lambda u: utilityP_inv(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk )
        self.uPinvP = lambda u: utilityP_invP(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk )
        self.uinvP = lambda u: utility_invP(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk)
        if self.vFuncBool:
            self.uinv = lambda u: utility_inv(u, n = self.N,v= self.v, varphi = self.varphi, gam=self.CRRA , theta = self.ThetaShk )
            
            
            
            
            
    def calc_EndOfPrdvP(self):
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

        EndOfPrdvP = []
        
        for i in range(len(self.ThetaShk)):

            
            def vp_next(shocks, a_nrm):
                
                return shocks[0] ** (-self.CRRA) \
                    * self.vPfuncNext[i](self.m_nrm_next(shocks, a_nrm))
    
    
    
            EndOfPrdvP_cond = (
                self.DiscFacEff
                * self.Rfree
                * self.PermGroFac ** (-self.CRRA)
                * calc_expectation(
                    self.IncShkDstn,
                    vp_next,
                    self.aNrmNow
                )
            )
            
            EndOfPrdvP.append(EndOfPrdvP_cond)
            
        
        EndOfPrdvP = np.array(EndOfPrdvP)
        
    
        
        return EndOfPrdvP

            
    def get_points_for_interpolation(self, EndOfPrdvP, aNrmNow):
        """
        Finds interpolation points (c,m) for the consumption function.
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        """
        cNrmNow_temp = self.uPinv(EndOfPrdvP)
        
        #for i in range(len(self.ThetaShk)):
            
            #cNrmNow.append(self.uPinv(EndOfPrdvP))
            #cNrmNow[i] = self.uPinv(EndOfPrdvP[i])
        
        
        mNrmNow_temp = []
        
        for i in range(len(self.ThetaShk )):
            
            mNrmNow_temp.append(cNrmNow_temp[i] + aNrmNow)
            
            
        mNrmNow= []
        cNrmNow= []
        
        for i in range(len(self.ThetaShk )):

           cNrmNow.append(np.insert(cNrmNow_temp[i], 0, 0.0, axis=-1))# Limiting consumption is zero as m approaches mNrmMin
           mNrmNow.append(np.insert(mNrmNow_temp[i], 0, self.BoroCnstNat, axis=-1))
            
        # Store these for calcvFunc
        self.cNrmNow = cNrmNow
        self.mNrmNow = mNrmNow
        
    
        
        print(cNrmNow)
        
        return cNrmNow, mNrmNow
    
    
    
    def use_points_for_interpolation(self, cNrm, mNrm, interpolator, ):
        """
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.
        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation.
        mNrm : np.array
            (Normalized) corresponding market resource points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.
        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        # Construct the unconstrained consumption function
        cFuncNowUnc=[]
        
        #print(np.shape(mNrm))
        
        for i in range(len(self.ThetaShk )):
            cFuncNowUnc.append(interpolator(mNrm[i], cNrm[i]))
            

        # Combine the constrained and unconstrained functions into the true consumption function
        # breakpoint()  # LowerEnvelope should only be used when BoroCnstArt is true
        
        cFuncNow  = []
        for i in range(len(self.ThetaShk )):
            cFuncNow.append(LowerEnvelope(cFuncNowUnc[i], self.cFuncNowCnst, nan_bool=False))
            

        # Make the marginal value function and the marginal marginal value function
        
        vPfuncNow=[]
        for i in range(len(self.ThetaShk )):
            vPfuncNow.append( MargValueFuncGHH(cFuncNow[i], self.CRRA, n = self.N, varphi = self.varphi, ThetaShk = self.ThetaShk[i] , v = self.v, mu = self.ThetaShk[0]))

        # Pack up the solution and return it
        
        solution_now = ConsumerSolution(
            cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow
        )

        return solution_now



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
                                                   "PermShkStd",
                                                   "Ghost",
                                                   
                                                    "PermShkCount",
                                                    "TranShkCount",
                                                    "TranShkStd",
                                                    "tax_rate",
                                                    "UnempPrb",
                                                    "IncUnemp",
                                                    "G",
                                                    
                                                    "v",
                                                    "varphi",
                                                    "ThetaShk",
    
                                                    
                                                  ]
    
    

    
    def __init__(self, cycles= 0, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 0, **kwds)

        solver = FBSNK_Solver
        self.solve_one_period = make_one_period_oo_solver(solver)


        
    
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = ((self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        
        TranShkDstnTEST = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        self.ThetaShk = np.insert(TranShkDstnTEST.X ,0, self.IncUnemp)


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
        self.IncShkDstnW = [combine_indep_dstns(PermShkDstnW,TranShkDstnW)]
        self.TranShkDstnW = [TranShkDstnW]
        self.PermShkDstnW = [PermShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
     
        
     
        TranShkDstnN     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnN.pmf  = np.insert(TranShkDstnN.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnN.X  = np.insert(TranShkDstnN.X*(((1.0-self.tax_rate)*(self.N + self.dx)*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnN     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnN= [combine_indep_dstns(PermShkDstnN,TranShkDstnN)]
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
        vF =[]
        vPF =[]
        vPPF =[]
        for i in range( len( self.ThetaShk)):
            vF.append(ValueFuncGHH(self.cFunc_terminal_, self.CRRA, n = self.N, varphi = self.varphi, ThetaShk = self.ThetaShk[i] , v = self.v, mu =self.ThetaShk[0] ))
            vPF.append(MargValueFuncGHH(self.cFunc_terminal_, self.CRRA,n = self.N, varphi = self.varphi, ThetaShk = self.ThetaShk[i] , v = self.v, mu =self.ThetaShk[0]))
            vPPF.append(MargMargValueFuncGHH(
            self.cFunc_terminal_, self.CRRA,n = self.N, varphi = self.varphi, ThetaShk = self.ThetaShk[i] , v = self.v,  mu =self.ThetaShk[0]
        ))
            
            
        self.solution_terminal.vFunc = vF
        self.solution_terminal.vPfunc = vPF
        self.solution_terminal.vPPfunc = vPPF
        
    
    
    
FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.048**.25,                       # Interest factor on assets
    "DiscFac": 0.97,                     # Intertemporal discount factor
    "LivPrb" : [.99375],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05, #.08                     # Probability of unemployment while working
    "IncUnemp" :  0.0954, #0.29535573122529635,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.16563445378151262,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "AgentCount" : 250000,                 # Number of agents of this type
    "T_sim" : 1400,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(1.3)-(.5**2)/2,# Mean of log initial assets
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
     "B" : 0,                               # Net Bond Supply
     "G" : .19, #.18
     "DisULabor": 38,
     "v": 2, 
     "varphi": 1
     }




#---------------------------------------------------------------------

ss_agent = FBSNK_agent(**FBSDict)
ss_agent.cycles = 0
ss_agent.dx = 0

ss_agent.solve()



    