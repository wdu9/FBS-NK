# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 02:28:50 2021

@author: wdu


python 3.8.8

econ-ark 0.11.0

numpy 1.20.2

matplotlib 3.4.1
"""
import numpy as np
from copy import copy, deepcopy
from HARK.interpolation import HARKinterpolator1D, MetricObject
from HARK.utilities import NullFunc




class ConsumerSolutionGHH(MetricObject):
    """
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.
    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over market
        resources: c = cFunc(m).
    vFunc : function
        The beginning-of-period value function for this period, defined over
        market resources: v = vFunc(m).
    vPfunc : function
        The beginning-of-period marginal value function for this period,
        defined over market resources: vP = vPfunc(m).
    vPPfunc : function
        The beginning-of-period marginal marginal value function for this
        period, defined over market resources: vPP = vPPfunc(m).
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.
    """

    distance_criteria = ["vPfunc"]

    def __init__(
        self,
        cFunc=None,
        vFunc=None,
        vPfunc=None,
        vPPfunc=None,
        mNrmMin=None,
        hNrm=None,
        MPCmin=None,
        MPCmax=None,
    ):
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def append_solution(self, new_solution):
        """
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in ConsMarkovModel, where we append solutions
        *conditional* on a particular value of a Markov state to each other in
        order to get the entire solution.
        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem; each attribute is a
            list representing state-conditional values or functions.
        Returns
        -------
        None
        """
        if type(self.cFunc) != list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert (
                NullFunc().distance(self.cFunc) == 0
            ), "append_solution called incorrectly!"

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc = [new_solution.cFunc]
            self.vFunc = [new_solution.vFunc]
            self.vPfunc = [new_solution.vPfunc]
            self.vPPfunc = [new_solution.vPPfunc]
            self.mNrmMin = [new_solution.mNrmMin]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)
            
            
            
#------------------------------------------------------------------------------------------------------



def GHHutility(c,n,v,varphi, gam, theta, mu):
    """
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter gam.
    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Utility
    Tests
    -----
    Test a value which should pass:
    >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    >>> utility(c=c, gam=gamma)
    -1.0
    """
    
    if theta == mu:
        return (c)** (1.0 - gam) / (1.0 - gam)
        
    else:
        
        Vn = (theta * varphi * (n**(1+v)) ) / (1+v)  
        
        return (c - Vn )** (1.0 - gam) / (1.0 - gam)
    
    #m = np.empty([len(theta), len(c)], dtype=float)
    
    #for i in range(len(theta)):
        #m[i] = (c - Vn[i] )** (1.0 - gam) / (1.0 - gam) 
        
       # if Vn[i] == mu:
            #m[i] = (c)** (1.0 - gam) / (1.0 - gam) 
    
    #if gam == 1:
        #return np.log(c)
    #else:
        #return m #(c - Vn )** (1.0 - gam) / (1.0 - gam)


def GHHutilityP(c,n,v,varphi, gam, theta, mu):
    """
    Evaluates constant relative risk aversion (CRRA) marginal utility of consumption
    c given risk aversion parameter gam.
    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Marginal utility
    """
    Vn = theta * (varphi/ (1.0 +v)) * (n**(1.0 +v))

    
    if theta == mu: 
        
        return (c  ) ** -gam 
    
    else:
        
    
        
        #m = np.empty([len(theta), len(c)], dtype=float)
        
        #for i in range(len(theta)):
            #m[i] = ( c - Vn[i] ) ** -gam
            
            #if Vn[i] == mu:
                #m[i] = c **- gam
                
        
        return  (c -  Vn) ** -gam  #m  (c -  Vn) ** -gam


def GHHutilityPP(c,n,v,varphi, gam, theta, mu):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal utility of
    consumption c given risk aversion parameter gam.
    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Marginal marginal utility
    """
    
    
    if theta == mu: 
        
        return  -gam * (c) ** (-gam - 1.0)
    
    else:
        
        Vn = theta * (varphi / (1+v)) * (n**(1+v))
        
        return -gam * (c - Vn) ** (-gam - 1.0)

    
   # m = np.empty([len(theta), len(c)], dtype=float)
    
    #for i in range(len(theta)):
        #m[i] = -gam *( c - Vn[i] ) ** (-gam - 1)
        
        #if Vn[i] == mu:
            #m[i] = -gam * (c) ** (-gam - 1.0)
    
    


def GHHutilityPPP(c,n,v,varphi, gam, theta, mu):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    utility of consumption c given risk aversion parameter gam.
    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal utility
    """
    
    if theta == mu:
        return (gam + 1.0) * gam * (c)** (-gam - 2.0)
        
    else:
        
        
        Vn = theta * (varphi/ (1+v)) * (n**(1+v))
    
    #m = np.empty([len(theta), len(c)], dtype=float)
    
    #for i in range(len(theta)):
        #m[i] = (gam + 1.0) * gam * (c - Vn[i])** (-gam - 2.0)
        
        #if Vn[i] == mu:
            #m[i] = (gam + 1.0) * gam * (c)** (-gam - 2.0)
    
    
        return (gam + 1.0) * gam * (c -Vn)** (-gam - 2.0)


def GHHutilityPPPP(c,n,v,varphi, gam, theta, mu):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    marginal utility of consumption c given risk aversion parameter gam.
    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal marginal utility
    """
    
    if theta == mu:
         return -(gam + 2.0) * (gam + 1.0) * gam * (c) ** (-gam - 3.0)
        
    else:
        
        Vn = theta * (varphi/ (1+v)) * (n**(1+v))
        return  -(gam + 2.0) * (gam + 1.0) * gam * (c - Vn) ** (-gam - 3.0)

    
    #m = np.empty([len(theta), len(c)], dtype=float)
    
    #for i in range(len(theta)):
        #m[i] = -(gam + 2.0) * (gam + 1.0) * gam * (c - Vn[i]) ** (-gam - 3.0)
        
        #if Vn[i] == mu:
            #m[i] = -(gam + 2.0) * (gam + 1.0) * gam * (c ) ** (-gam - 3.0)
    




def GHHutility_inv(u,n,v,varphi, gam, theta):
    """
    Evaluates the inverse of the CRRA utility function (with risk aversion para-
    meter gam) at a given utility level u.
    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given utility value
    """
    
    Vn = theta * (varphi/ (1+v)) * (n**(1+v))
    Vn[0] = 0
    
    m = u - Vn.reshape((len(Vn),1))
    
    if gam == 1:
        return np.exp(u)
    else:
        return m #(((1.0 - gam) * u) ** (1 / (1.0 - gam))) + Vn


def GHHutilityP_inv(uP,n,v,varphi, gam, theta):
    """
    Evaluates the inverse of the CRRA marginal utility function (with risk aversion
    parameter gam) at a given marginal utility level uP.
    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value.
    """
    
    Vn = theta * (varphi/ (1+v)) * (n**(1+v))
    Vn[0] = 0
    
    m  =  (uP ** (-1.0 / gam))  + Vn.reshape((len(Vn),1))
         
    return m #(uP ** (-1.0 / gam)) + Vn


def GHHutility_invP(u,n,v,varphi, gam, theta):
    """
    Evaluates the derivative of the inverse of the CRRA utility function (with
    risk aversion parameter gam) at a given utility level u.
    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given utility value
    """
    Vn = theta * (varphi/ (1+v)) * (n**(1+v))
    Vn[0]=0
        
   
    m = (((1.0 - gam) * u) ** (gam / (1.0 - gam)))  + Vn.reshape((len(Vn),1))
        
   
            
    if gam == 1:
        return np.exp(u)
    else:
        return m # (((1.0 - gam) * u) ** (gam / (1.0 - gam))) + Vn


def GHHutilityP_invP(uP,n,v,varphi, gam,theta):
    """
    Evaluates the derivative of the inverse of the CRRA marginal utility function
    (with risk aversion parameter gam) at a given marginal utility level uP.
    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion
    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given marginal utility value
    """
    
    Vn = theta * (varphi/ (1+v)) * (n**(1+v))
    Vn[0]=0

    
    m = ((-1.0 / gam) * uP ** (-1.0 / gam - 1.0)) + Vn.reshape((len(Vn),1))
        
    
    return m #((-1.0 / gam) * uP ** (-1.0 / gam - 1.0)) + Vn






#----------------------------------------------------------------------------------



class ValueFuncGHH(MetricObject):
    """
    A class for representing a value function.  The underlying interpolation is
    in the space of (state,u_inv(v)); this class "re-curves" to the value function.
    Parameters
    ----------
    vFuncNvrs : function
        A real function representing the value function composed with the
        inverse utility function, defined on the state: u_inv(vFunc(state))
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["func", "CRRA"]

    def __init__(self, vFuncNvrs, CRRA,n,v,varphi,ThetaShk, mu):
        self.func = deepcopy(vFuncNvrs)
        self.CRRA = CRRA
        self.v = v
        self.varphi = varphi
        self.ThetaShk = ThetaShk
        self.n = n
        self.mu = mu

    def __call__(self, *vFuncArgs):
        """
        Evaluate the value function at given levels of market resources m.
        Parameters
        ----------
        vFuncArgs : floats or np.arrays, all of the same dimensions.
            Values for the state variables. These usually start with 'm',
            market resources normalized by the level of permanent income.
        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with the given states; has
            same size as the state inputs.
        """
        return GHHutility(self.func(*vFuncArgs), gam=self.CRRA, n = self.n, varphi = self.varphi, theta = self.ThetaShk , v = self.v, mu=self.mu)

class MargValueFuncGHH(MetricObject):
    """
    A class for representing a marginal value function in models where the
    standard envelope condition of dvdm(state) = u'(c(state)) holds (with CRRA utility).
    Parameters
    ----------
    cFunc : function.
        Its first argument must be normalized market resources m.
        A real function representing the marginal value function composed
        with the inverse marginal utility function, defined on the state
        variables: uP_inv(dvdmFunc(state)).  Called cFunc because when standard
        envelope condition applies, uP_inv(dvdm(state)) = cFunc(state).
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA, v, varphi, n, ThetaShk,mu):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        self.v = v
        self.varphi = varphi
        self.ThetaShk = ThetaShk
        self.n = n
        self.mu = mu

        
        

    def __call__(self, *cFuncArgs):
        """
        Evaluate the marginal value function at given levels of market resources m.
        Parameters
        ----------
        cFuncArgs : floats or np.arrays
            Values of the state variables at which to evaluate the marginal
            value function.
        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with state
            cFuncArgs
        """
        return GHHutilityP( self.cFunc(*cFuncArgs), gam=self.CRRA , n = self.n, varphi = self.varphi, theta = self.ThetaShk , v = self.v, mu=self.mu)

    def derivativeX(self, *cFuncArgs):
        """
        Evaluate the derivative of the marginal value function with respect to
        market resources at given state; this is the marginal marginal value
        function.
        Parameters
        ----------
        cFuncArgs : floats or np.arrays
            State variables.
        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with
            state cFuncArgs; has same size as inputs.
        """

        # The derivative method depends on the dimension of the function
        if isinstance(self.cFunc, (HARKinterpolator1D)):
            c, MPC = self.cFunc.eval_with_derivative(*cFuncArgs)

        elif hasattr(self.cFunc, 'derivativeX'):
            c = self.cFunc(*cFuncArgs)
            MPC = self.cFunc.derivativeX(*cFuncArgs)

        else:
            raise Exception(
                "cFunc does not have a 'derivativeX' attribute. Can't compute"
                + "marginal marginal value."
            )

        return MPC * GHHutilityPP(c, gam=self.CRRA , n = self.n, varphi = self.varphi, theta = self.ThetaShk , v = self.v, mu = self.mu)
    
    
    
    
    
    
    
class MargMargValueFuncGHH(MetricObject):
    """
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of dvdm = u'(c(state)) holds (with CRRA utility).
    Parameters
    ----------
    cFunc : function.
        Its first argument must be normalized market resources m.
        A real function representing the marginal value function composed
        with the inverse marginal utility function, defined on the state
        variables: uP_inv(dvdmFunc(state)).  Called cFunc because when standard
        envelope condition applies, uP_inv(dvdm(state)) = cFunc(state).
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA, v, varphi, n, ThetaShk,mu):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        self.v = v
        self.varphi = varphi
        self.ThetaShk = ThetaShk
        self.n = n
        self.mu = mu

    def __call__(self, *cFuncArgs):
        """
        Evaluate the marginal marginal value function at given levels of market
        resources m.
        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.
        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        """

        # The derivative method depends on the dimension of the function
        if isinstance(self.cFunc, (HARKinterpolator1D)):
            c, MPC = self.cFunc.eval_with_derivative(*cFuncArgs)

        elif hasattr(self.cFunc, 'derivativeX'):
            c = self.cFunc(*cFuncArgs)
            MPC = self.cFunc.derivativeX(*cFuncArgs)

        else:
            raise Exception(
                "cFunc does not have a 'derivativeX' attribute. Can't compute"
                + "marginal marginal value."
            )

        return MPC * GHHutilityPP(c, gam=self.CRRA, n = self.n, varphi = self.varphi, theta = self.ThetaShk , v = self.v, mu = self.mu)
    