# -*- coding: utf-8 -*-

from __future__ import print_function
import sys 
import os
from copy import copy, deepcopy
import numpy as np
import scipy as sc
import numba as nb
from numba import jit
from scipy import sparse as sp
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.utilities import make_grid_exp_mult
from JAC_Utility import DiscreteDistribution2, combine_indep_dstns2
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform, calc_expectation
import matplotlib.pyplot as plt


"""
Created on Sun Jul  4 15:02:16 2021

@author: wdu
"""
'''
Extends the IndShockConsumerType agent to store a distribution of agents and
calculates a transition matrix for this distribution, along with the steady
state distribution
'''

class JACTran(IndShockConsumerType):
    '''
    An extension of the IndShockConsumerType that adds methods to handle
    the distribution of agents over market resources and permanent income.
    These methods could eventually become part of IndShockConsumterType itself
    '''        
    
    
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
                                                   "jacT",
                                                   "PermShkStd",
                                                   "Ghost",
                                                   
                                                    "PermShkCount",
                                                    "TranShkCount",
                                                    "TranShkStd",
                                                    "tax_rate",
                                                    "UnempPrb",
                                                    "IncUnemp",
                                                    "G",
                                                    "DisULabor",
                                                    "InvFrisch",
                                                    "s",
    
                                                    
                                                  ]

    #def __init__(self,cycles=0,time_flow=True,**kwds):
    def __init__(self,cycles=0,**kwds):

        '''
        Just calls on IndShockConsumperType
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        
        IndShockConsumerType.__init__(self, cycles = 0, **kwds)
        ## Initialize an IndShockConsumerType
        #IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)
        
        
    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        
        if type(self.Rfree) == list:
             self.N = ((self.IncUnemp*self.UnempPrb ) + self.G + (1 - (1/(self.Rfree[0]) ) ) * self.B) / (self.wage*self.tax_rate)
            
        else:
            
            self.N = ((self.IncUnemp*self.UnempPrb ) + self.G + (1 - (1/(self.Rfree) ) ) * self.B) / (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        

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
        
        
             
        TranShkDstnt    = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnt.pmf  = np.insert(TranShkDstnt.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnt.X  = np.insert(TranShkDstnt.X*(((1.0- (self.tax_rate +self.dx)) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnt     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnt= [combine_indep_dstns2(PermShkDstnt,TranShkDstnt)]
        self.TranShkDstnt = [TranShkDstnt]
        self.PermShkDstnt = [PermShkDstnt]
        self.add_to_time_vary('IncShkDstnt')
        
        TranShkDstnP     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnP.pmf  = np.insert(TranShkDstnP.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnP.X  = np.insert(TranShkDstnP.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnP     = MeanOneLogNormal(self.PermShkStd[0] + self.dx ,123).approx(self.PermShkCount)
        self.IncShkDstnP= [combine_indep_dstns2(PermShkDstnP,TranShkDstnP)]
        self.TranShkDstnP = [TranShkDstnP]
        self.PermShkDstnP = [PermShkDstnP]
        self.add_to_time_vary('IncShkDstnP')
        
        TranShkDstnT     = MeanOneLogNormal(self.TranShkStd[0] + self.dx,123).approx(self.TranShkCount)
        TranShkDstnT.pmf  = np.insert(TranShkDstnT.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnT.X  = np.insert(TranShkDstnT.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnT     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnT= [combine_indep_dstns2(PermShkDstnT,TranShkDstnT)]
        self.TranShkDstnT = [TranShkDstnT]
        self.PermShkDstnT = [PermShkDstnT]
        self.add_to_time_vary('IncShkDstnT')
        
        
    def DefineDistributionGrid(self, Dist_mGrid=None, Dist_pGrid=None):
        '''
        Defines the grid on which the distribution is defined
        
        Parameters
        ----------
        Dist_mGrid : np.array()
            Grid for distribution over normalized market resources
        Dist_pGrid : np.array()
            Grid for distribution over permanent income
        
        Returns
        -------
        None
        '''  
        #if self.cycles != 0:
            #print('Distributional methods presently only work for perpetual youth agents (cycles=0)')
        if self.cycles==0:
            if Dist_mGrid == None:
                self.Dist_mGrid = self.aXtraGrid
            else:
                self.Dist_mGrid = Dist_mGrid
            if Dist_pGrid == None:
                num_points = 50
                #Dist_pGrid is taken to cover most of the ergodic distribution
                p_variance = self.PermShkStd[0]**2
                max_p = 20.0*(p_variance/(1-self.LivPrb[0]))**0.5
                one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2)
                self.Dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid)
            else:
                self.Dist_pGrid = Dist_pGrid
                
            
                
        elif self.T_cycle !=0:#and self.cycles!=0:
            
            
            
                
            if Dist_mGrid == None:
                self.Dist_mGrid = self.aXtraGrid
            else:
                self.Dist_mGrid = Dist_mGrid
            if Dist_pGrid == None:
                num_points = 50
                #Dist_pGrid is taken to cover most of the ergodic distribution
                p_variance = self.PermShkStd[0]**2
                max_p = 20.0*(p_variance/(1-self.LivPrb[0]))**0.5
                one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2)
                self.Dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid)
            else:
                self.Dist_pGrid = Dist_pGrid
            
     
            
                
    def CalcTransitionMatrix(self):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next
        ''' 
        
        
        if self.cycles == 0: 
            Dist_mGrid = self.Dist_mGrid
            Dist_pGrid = self.Dist_pGrid
            aNext = Dist_mGrid - self.solution[0].cFunc(Dist_mGrid)
            
            
            self.Cnow = self.solution[0].cFunc(Dist_mGrid)
            
            
            bNext = self.Rfree*aNext
            ShockProbs = self.IncShkDstn[0].pmf
            TranShocks = self.IncShkDstn[0].X[1]
            PermShocks = self.IncShkDstn[0].X[0]
            LivPrb = self.LivPrb[0]
            #New borns have this distribution (assumes start with no assets and permanent income=1)
            NewBornDist = self.JumpToGrid(TranShocks,np.ones_like(TranShocks),ShockProbs)
            TranMatrix = np.zeros((len(Dist_mGrid)*len(Dist_pGrid),len(Dist_mGrid)*len(Dist_pGrid)))
            for i in range(len(Dist_mGrid)):
                for j in range(len(Dist_pGrid)):
                    mNext_ij = bNext[i]/PermShocks + TranShocks
                    pNext_ij = Dist_pGrid[j]*PermShocks
                    TranMatrix[:,i*len(Dist_pGrid)+j] = LivPrb*self.JumpToGrid(mNext_ij, pNext_ij, ShockProbs) + (1.0-LivPrb)*NewBornDist
            self.TranMatrix = TranMatrix
        
        
        elif self.T_cycle!=0 and self.cycles!=0: 
            
            self.CNrmList = []
            self.TranMatList =[]
            
            
            Dist_mGrid = self.Dist_mGrid
            Dist_pGrid = self.Dist_pGrid
            
            for i in range(self.T_cycle): 
            
                aNext = Dist_mGrid - self.solution[i].cFunc(Dist_mGrid)
                
                self.Cnow = self.solution[i].cFunc(Dist_mGrid)
                self.CNrmList.append(self.Cnow)
               
                bNext = self.Rfree[i]*aNext
                ShockProbs = self.IncShkDstn[i].pmf
                TranShocks = self.IncShkDstn[i].X[1]
                PermShocks = self.IncShkDstn[i].X[0]
                LivPrb = self.LivPrb[i]
                #New borns have this distribution (assumes start with no assets and permanent income=1)
                NewBornDist = self.JumpToGrid(TranShocks,np.ones_like(TranShocks),ShockProbs)
                TranMatrix = np.zeros((len(Dist_mGrid)*len(Dist_pGrid),len(Dist_mGrid)*len(Dist_pGrid)))
                for i in range(len(Dist_mGrid)):
                    for j in range(len(Dist_pGrid)):
                        mNext_ij = bNext[i]/PermShocks + TranShocks
                        pNext_ij = Dist_pGrid[j]*PermShocks
                        TranMatrix[:,i*len(Dist_pGrid)+j] = LivPrb*self.JumpToGrid(mNext_ij, pNext_ij, ShockProbs) + (1.0-LivPrb)*NewBornDist
                
                #self.TranMatrix = TranMatrix
                
            
                self.TranMatList.append(TranMatrix)
                print(len(self.TranMatList))
            
            
        
        
    def JumpToGrid(self,m_vals, perm_vals, probs):
        '''
        Distributes values onto a predefined grid, maintaining the means
        ''' 
        probGrid = np.zeros((len(self.Dist_mGrid),len(self.Dist_pGrid)))
        mIndex = np.digitize(m_vals,self.Dist_mGrid) - 1
        mIndex[m_vals <= self.Dist_mGrid[0]] = -1
        mIndex[m_vals >= self.Dist_mGrid[-1]] = len(self.Dist_mGrid)-1
        
        pIndex = np.digitize(perm_vals,self.Dist_pGrid) - 1
        pIndex[perm_vals <= self.Dist_pGrid[0]] = -1
        pIndex[perm_vals >= self.Dist_pGrid[-1]] = len(self.Dist_pGrid)-1
        
        for i in range(len(m_vals)):
            if mIndex[i]==-1:
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(self.Dist_mGrid)-1:
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else:
                mlowerIndex = mIndex[i]
                mupperIndex = mIndex[i]+1
                mlowerWeight = (self.Dist_mGrid[mupperIndex]-m_vals[i])/(self.Dist_mGrid[mupperIndex]-self.Dist_mGrid[mlowerIndex])
                mupperWeight = 1.0 - mlowerWeight
                
            if pIndex[i]==-1:
                plowerIndex = 0
                pupperIndex = 0
                plowerWeight = 1.0
                pupperWeight = 0.0
            elif pIndex[i]==len(self.Dist_pGrid)-1:
                plowerIndex = -1
                pupperIndex = -1
                plowerWeight = 1.0
                pupperWeight = 0.0
            else:
                plowerIndex = pIndex[i]
                pupperIndex = pIndex[i]+1
                plowerWeight = (self.Dist_pGrid[pupperIndex]-perm_vals[i])/(self.Dist_pGrid[pupperIndex]-self.Dist_pGrid[plowerIndex])
                pupperWeight = 1.0 - plowerWeight
                
            probGrid[mlowerIndex][plowerIndex] = probGrid[mlowerIndex][plowerIndex] + probs[i]*mlowerWeight*plowerWeight
            probGrid[mlowerIndex][pupperIndex] = probGrid[mlowerIndex][pupperIndex] + probs[i]*mlowerWeight*pupperWeight
            probGrid[mupperIndex][plowerIndex] = probGrid[mupperIndex][plowerIndex] + probs[i]*mupperWeight*plowerWeight
            probGrid[mupperIndex][pupperIndex] = probGrid[mupperIndex][pupperIndex] + probs[i]*mupperWeight*pupperWeight
            
        return probGrid.flatten()
    
    
    def CalcErgodicDist(self):
        
        '''
        Calculates the egodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is reshaped as an array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.
        ''' 
        
        eigen, ergodic_distr = sp.linalg.eigs(self.TranMatrix , k=1 , which='LM')
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        self.vec_dstn = ergodic_distr
        self.ergodic_distr = ergodic_distr.reshape((len(self.Dist_mGrid),len(self.Dist_pGrid)))







FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.05**.25,                       # Interest factor on assets
    "DiscFac": 0.96,                     # Intertemporal discount factor
    "LivPrb" : [.99375],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [.03], #[(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05, #.08                     # Probability of unemployment while working
    "IncUnemp" :  .3, #0.0954, #0.29535573122529635,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : .2, #0.16563445378151262,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "AgentCount" : 50000,                 # Number of agents of this type
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(1.6)-(.5**2)/2,# Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.3, 
     "dx"         : 0,                  #Deviation from steady state
     "jac"        : False,
     "jacW"       : False, 
     "jacN"       : False,
     "jact"       : False,
     "jacT"       : False,
     "jacPerm"    : False,
     "Ghost"      : False, 
     
     
    #New Economy Parameters
     "SSWmu" : 1.05 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.012,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : .2,                               # Net Bond Supply
     "G" : .3,
     "DisULabor": 0.8823685356415617,
     "InvFrisch": 2 ,
     "s" : 1
     }



G=.4
t=.2 #0.16563445378151262
Inc = .2
mho=.06
r = (1.05)**.25 - 1 
B=.2 #.65

w = (1/1.012)
N = (Inc*mho + G  + (1 - (1/(1+r)) ) *B) / (w*t) 
q = ((1-w)*N)/r

A = ( B/(1+r) ) + q

#print(N)
#print(q)
#print(A)

ss = JACTran(**FBSDict)
ss.cycles=0
ss.T_sim = 1000
ss.solve()
ss.initialize_sim()
ss.simulate()


AggC = np.mean((ss.state_now['mNrm'] - ss.state_now['aNrm'])*ss.state_now['pLvl'])
CNORM = np.mean(ss.state_now['mNrm'] - ss.state_now['aNrm'])
print(AggC)
print(CNORM)




ss.DefineDistributionGrid()
ss.CalcTransitionMatrix()
ss.CalcErgodicDist()

vecDstn = ss.vec_dstn
erg_Dstn = ss.ergodic_distr
TranMat = ss.TranMatrix

Ca = ss.Cnow
p = ss.Dist_pGrid
c = ss.Cnow
grid = np.zeros((len(c),len(p)))

for j in range(len(p)):
    grid[:,j] = p[j]*c
    
C = np.dot(grid.flatten(),vecDstn)
print(C)











class FBSNK_JAC(JACTran):
    


    
  
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

    

     def CalcErgodicDist(self):
        
        '''
        Calculates the egodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is reshaped as an array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.
        ''' 
        
        eigen, ergodic_distr = sp.linalg.eigs(ss.TranMatrix , k=1 , which='LM')
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        self.vec_dstn = ergodic_distr
        self.ergodic_distr = ergodic_distr.reshape((len(self.Dist_mGrid),len(self.Dist_pGrid)))










params = deepcopy(FBSDict)
params['T_cycle']= 30
params['LivPrb']= params['T_cycle']*[ss.LivPrb[0]]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[ss.PermShkStd[0]]
params['TranShkStd']= params['T_cycle']*[ss.TranShkStd[0]]
params['Rfree'] = params['T_cycle']*[ss.Rfree]



example = FBSNK_JAC(**params)
example.pseudo_terminal = False
example.cycles = 1

example.jac= True

if example.jac == True:
    example.dx = .1
    example.del_from_time_inv('Rfree')
    example.add_to_time_vary('Rfree')
    example.IncShkDstn = params['T_cycle']*ss.IncShkDstn
    
    
i = 15
if example.jac == True:
    example.Rfree = (i)*[ss.Rfree] + [ss.Rfree + example.dx] + (params['T_cycle'] - i - 1)*[ss.Rfree]
    
    
    
    
example.solve()



example.DefineDistributionGrid()
example.CalcTransitionMatrix()
example.CalcErgodicDist()








Distributions = []

AggCNorm_List = []

AggC_List =[]

T=params['T_cycle']

a = vecDstn

for i in range(T):
    
    
    #Distributions.append(np.dot(example.TranMatList[i],a))
    #a = np.dot(example.TranMatList[i],a)

    p = example.Dist_pGrid
    c = example.CNrmList[i]
    
    grid = np.zeros((len(c),len(p)))

    for j in range(len(p)):
        grid[:,j] = p[j]*c
        
    C = np.dot(grid.flatten(),a)
    
    AggC_List.append(C)
    
    a = np.dot(example.TranMatList[i],a)

AggC_List  = np.array(AggC_List).T[0]

plt.plot(AggC_List)
plt.show()
    






'''


if __name__ == '__main__':
    import HARK.ConsumptionSaving.ConsumerParameters as Params
    from HARK.utilities import plot_funcs_der, plot_funcs
    import time
    mystr = lambda number : "{:.4f}".format(number)
    
    
    # Make and solve an example consumer with idiosyncratic income shocks
    IndShock_extendExample = IndShockConsumerType_extend(**Params.init_idiosyncratic_shocks)
    IndShock_extendExample.cycles = 0 # Make this type have an infinite horizon

    start_time = clock()
    IndShock_extendExample.solve()
    end_time = clock()
    print('Solving a consumer with idiosyncratic shocks took ' + mystr(end_time-start_time) + ' seconds.')
    
    IndShock_extendExample.DefineDistributionGrid()   
    start_time = clock()
    IndShock_extendExample.CalcTransitionMatrix()
    end_time = clock()
    print('Calculating the transition matrix took ' + mystr(end_time-start_time) + ' seconds.')
    start_time = clock()
    IndShock_extendExample.CalcErgodicDist()
    end_time = clock()
    print('Calculating the ergodic distribution took ' + mystr(end_time-start_time) + ' seconds.')
    
    
    
    
    
A = np.array ([[0,1] , [-2,-3]])
    
    
eigenA, dA = sp.linalg.eigs(A)

eigenA =eigenA.real

print('space')
print(dA)
print(dA[:,0])
print(np.dot(A,dA[:,0]))


'''