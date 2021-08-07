

from __future__ import print_function
import sys 
import os
from copy import copy, deepcopy
import numpy as np
import scipy as sc
from scipy import sparse as sp
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, construct_assets_grid
from HARK.utilities import make_grid_exp_mult
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform, calc_expectation
import matplotlib.pyplot as plt
import time

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

        TranShkDstn     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstn.pmf  = np.insert(TranShkDstn.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)   
        TranShkDstn.X  = np.insert(TranShkDstn.X*(1.0-self.tax_rate),0,self.IncUnemp)
        PermShkDstn     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        
        PermShk_ntrl_msr = deepcopy(PermShkDstn)
        PermShk_ntrl_msr.pmf = PermShk_ntrl_msr.X*PermShk_ntrl_msr.pmf
        self.IncShkDstn = [combine_indep_dstns(PermShkDstn,TranShkDstn)]
        self.IncShkDstn_ntrl_msr = [combine_indep_dstns(PermShk_ntrl_msr,TranShkDstn)]
        self.TranShkDstn = [TranShkDstn]
        self.PermShkDstn = [PermShkDstn]
        self.add_to_time_vary('IncShkDstn')
        

       
    def Define_Distribution_Grid(self, Dist_mGrid=None, Dist_pGrid=None):
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
        
        
 
        if self.cycles == 0:
            if Dist_mGrid == None:
                self.Dist_mGrid = self.aXtraGrid
            else:
                self.Dist_mGrid = Dist_mGrid #If grid of market resources prespecified then use as mgrid
            if Dist_pGrid == None:
                num_points = 50
                #Dist_pGrid is taken to cover most of the ergodic distribution
                p_variance = self.PermShkStd[0]**2 #set variance of permanent income shocks
                max_p = 20.0*(p_variance/(1-self.LivPrb[0]))**0.5 # Consider probability of staying alive
                one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2)
                self.Dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid) #Compute permanent income grid
            else:
                self.Dist_pGrid = Dist_pGrid #If grid of permanent income prespecified then use as pgrid
        
        elif self.T_cycle != 0:
            
            if Dist_mGrid == None:
                self.Dist_mGrid = self.aXtraGrid
            else:
                self.Dist_mGrid = Dist_mGrid #If grid of market resources prespecified then use as mgrid
                    
            if Dist_pGrid == None:
                
                self.Dist_pGrid = [] #list of grids of permanent income    
                
                for i in range(self.T_cycle):
                    
                    num_points = 50
                    #Dist_pGrid is taken to cover most of the ergodic distribution
                    p_variance = self.PermShkStd[i]**2 # set variance of permanent income shocks this period
                    max_p = 20.0*(p_variance/(1-self.LivPrb[i]))**0.5 # Consider probability of staying alive this period
                    one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2)
                    
                    Dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid) # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    self.Dist_pGrid.append(Dist_pGrid)

            else:
                self.Dist_pGrid = Dist_pGrid #If grid of permanent income prespecified then use as pgrid
                
                
            
                
    def Calc_Transition_Matrix(self):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset grids. 
        ''' 
        
        
        if self.cycles == 0: 
            Dist_mGrid = self.Dist_mGrid
            Dist_pGrid = self.Dist_pGrid
            aNext = Dist_mGrid - self.solution[0].cFunc(Dist_mGrid)
            
            self.aPolGrid = aNext # Steady State Asset Policy Grid
            self.cPolGrid = self.solution[0].cFunc(Dist_mGrid) #Steady State Consumption Policy Grid
            
            # Obtain shocks and shock probabilities from income distribution
            bNext = self.Rfree*aNext # Bank Balances next period (Interest rate * assets)
            ShockProbs = self.IncShkDstn[0].pmf  # Probability of shocks 
            TranShocks = self.IncShkDstn[0].X[1] # Transitory shocks
            PermShocks = self.IncShkDstn[0].X[0] # Permanent shocks
            LivPrb = self.LivPrb[0] # Update probability of staying alive
            
            #New borns have this distribution (assumes start with no assets and permanent income=1)
            NewBornDist = self.Jump_To_Grid(TranShocks,np.ones_like(TranShocks),ShockProbs,Dist_mGrid,Dist_pGrid)
            
            # Generate Steady State Transition Matrix
            TranMatrix = np.zeros((len(Dist_mGrid)*len(Dist_pGrid),len(Dist_mGrid)*len(Dist_pGrid)))
            for i in range(len(Dist_mGrid)):
                for j in range(len(Dist_pGrid)):
                    mNext_ij = bNext[i]/PermShocks + TranShocks # Compute next period's market resources given todays bank balances bnext[i]
                    pNext_ij = Dist_pGrid[j]*PermShocks # Computes next period's permanent income level by applying permanent income shock
                    TranMatrix[:,i*len(Dist_pGrid)+j] = LivPrb*self.Jump_To_Grid(mNext_ij, pNext_ij, ShockProbs,Dist_mGrid,Dist_pGrid) + (1.0-LivPrb)*NewBornDist
            self.TranMatrix = TranMatrix
            
        elif self.T_cycle!= 0:
            
            self.cPolGrid = [] # List of consumption policy Grids for each period in T_cycle
            self.aPolGrid = [] # List of asset policy grids for each period in T_cycle
            self.TranMatrix = [] # List of transition matrices
            self.TranMatrix_M =[]
            Dist_mGrid =  self.Dist_mGrid
            
            if  type(self.Dist_pGrid) == list:
                pGrid =  self.Dist_pGrid
            else:
                pGrid =  self.Dist_pGrid

            for k in range(self.T_cycle):
                                
                Dist_pGrid = pGrid[k]# Permanent income grid this period
                
                Cnow = self.solution[k].cFunc(Dist_mGrid) #Consumption policy grid in period k
                self.cPolGrid.append(Cnow) #Add to list

                aNext = Dist_mGrid - Cnow # Asset policy grid in period k
                self.aPolGrid.append(aNext) # Add to list
                
                bNext = self.Rfree[k]*aNext # Update interest rate this period
                
                #Obtain shocks and shock probabilities from income distribution this period
                ShockProbs = self.IncShkDstn[k].pmf  #probability of shocks this period
                TranShocks = self.IncShkDstn[k].X[1] #Transitory shocks this period
                PermShocks = self.IncShkDstn[k].X[0] #Permanent shocks this period
                LivPrb = self.LivPrb[k] # Update probability of staying alive this period
                
                #New borns have this distribution (assumes start with no assets and permanent income=1)
                NewBornDist = self.Jump_To_Grid(TranShocks,np.ones_like(TranShocks),ShockProbs,Dist_mGrid,Dist_pGrid)
                
                # Generate Transition Matrix this period
                TranMatrix = np.zeros((len(Dist_mGrid)*len(Dist_pGrid),len(Dist_mGrid)*len(Dist_pGrid))) 
                for i in range(len(Dist_mGrid)):
                    for j in range(len(Dist_pGrid)):
                        mNext_ij = bNext[i]/PermShocks + TranShocks # Compute next period's market resources given todays bank balances bnext[i]
                        pNext_ij = Dist_pGrid[j]*PermShocks # Computes next period's permanent income level by applying permanent income shock
                        TranMatrix[:,i*len(Dist_pGrid)+j] = LivPrb*self.Jump_To_Grid(mNext_ij, pNext_ij, ShockProbs,Dist_mGrid,Dist_pGrid) + (1.0-LivPrb)*NewBornDist 
                TranMatrix = TranMatrix
                self.TranMatrix.append(TranMatrix)
                         

        
    def Calc_Transition_Matrix_M(self):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset grids. 
        ''' 
        
        
        if self.cycles == 0: 
            Dist_mGrid = self.Dist_mGrid
            Dist_pGrid = self.Dist_pGrid
            aNext = Dist_mGrid - self.solution[0].cFunc(Dist_mGrid)
            
            self.aPolGrid = aNext # Steady State Asset Policy Grid
            self.cPolGrid = self.solution[0].cFunc(Dist_mGrid) #Steady State Consumption Policy Grid
            
            # Obtain shocks and shock probabilities from income distribution
            bNext = self.Rfree*aNext # Bank Balances next period (Interest rate * assets)

            ShockProbs_ntrl = self.IncShkDstn_ntrl_msr[0].pmf  #probability of shocks this period
            TranShocks_ntrl = self.IncShkDstn_ntrl_msr[0].X[1] #Transitory shocks this period
            PermShocks_ntrl = self.IncShkDstn_ntrl_msr[0].X[0] #Permanent shocks this period
                
            LivPrb = self.LivPrb[0] # Update probability of staying alive
            
            #New borns have this distribution (assumes start with no assets and permanent income=1)
            NewBornDist = self.Jump_To_Grid_M(TranShocks_ntrl,ShockProbs_ntrl,Dist_mGrid)
            
            # Generate Steady State Transition Matrix
            TranMatrix_M = np.zeros((len(Dist_mGrid),len(Dist_mGrid))) 
            for i in range(len(Dist_mGrid)):
                    mNext_ij = bNext[i]/PermShocks_ntrl + TranShocks_ntrl # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_M[:,i] = LivPrb*self.Jump_To_Grid_M(mNext_ij, ShockProbs_ntrl,Dist_mGrid) + (1.0-LivPrb)*NewBornDist 
            self.TranMatrix_M = TranMatrix_M
        
        elif self.T_cycle!= 0:
            
            self.cPolGrid = [] # List of consumption policy Grids for each period in T_cycle
            self.aPolGrid = [] # List of asset policy grids for each period in T_cycle
            self.TranMatrix_M =[]
            Dist_mGrid =  self.Dist_mGrid


            for k in range(self.T_cycle):
                                                
                Cnow = self.solution[k].cFunc(Dist_mGrid) #Consumption policy grid in period k
                self.cPolGrid.append(Cnow) #Add to list

                aNext = Dist_mGrid - Cnow # Asset policy grid in period k
                self.aPolGrid.append(aNext) # Add to list
                
                bNext = self.Rfree[k]*aNext # Update interest rate this period
                
                LivPrb = self.LivPrb[k] # Update probability of staying alive this period
                

                ShockProbs_ntrl = self.IncShkDstn_ntrl_msr[k].pmf  #probability of shocks this period
                TranShocks_ntrl = self.IncShkDstn_ntrl_msr[k].X[1] #Transitory shocks this period
                PermShocks_ntrl = self.IncShkDstn_ntrl_msr[k].X[0] #Permanent shocks this period
                
                
                #New borns have this distribution (assumes start with no assets and permanent income=1)
                NewBornDist = self.Jump_To_Grid_M(TranShocks_ntrl,ShockProbs_ntrl,Dist_mGrid)
                
                # Generate Transition Matrix this period
                TranMatrix_M = np.zeros((len(Dist_mGrid),len(Dist_mGrid))) 
                for i in range(len(Dist_mGrid)):
                        mNext_ij = bNext[i]/PermShocks_ntrl + TranShocks_ntrl # Compute next period's market resources given todays bank balances bnext[i]
                        TranMatrix_M[:,i] = LivPrb*self.Jump_To_Grid_M(mNext_ij, ShockProbs_ntrl,Dist_mGrid) + (1.0-LivPrb)*NewBornDist 
                TranMatrix_M = TranMatrix_M
                self.TranMatrix_M.append(TranMatrix_M)
                
                
                
    def Jump_To_Grid(self,m_vals, perm_vals, probs ,Dist_mGrid,Dist_pGrid ):
        '''
        Distributes values onto a predefined grid, maintaining the means.
        ''' 
    
        probGrid = np.zeros((len(Dist_mGrid),len(Dist_pGrid)))
        mIndex = np.digitize(m_vals,Dist_mGrid) - 1
        mIndex[m_vals <= Dist_mGrid[0]] = -1
        mIndex[m_vals >= Dist_mGrid[-1]] = len(Dist_mGrid)-1
        
        pIndex = np.digitize(perm_vals,Dist_pGrid) - 1
        pIndex[perm_vals <= Dist_pGrid[0]] = -1
        pIndex[perm_vals >= Dist_pGrid[-1]] = len(Dist_pGrid)-1
        
        for i in range(len(m_vals)):
            if mIndex[i]==-1:
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(Dist_mGrid)-1:
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else:
                mlowerIndex = mIndex[i]
                mupperIndex = mIndex[i]+1
                mlowerWeight = (Dist_mGrid[mupperIndex]-m_vals[i])/(Dist_mGrid[mupperIndex]-Dist_mGrid[mlowerIndex])
                mupperWeight = 1.0 - mlowerWeight
                
            if pIndex[i]==-1:
                plowerIndex = 0
                pupperIndex = 0
                plowerWeight = 1.0
                pupperWeight = 0.0
            elif pIndex[i]==len(Dist_pGrid)-1:
                plowerIndex = -1
                pupperIndex = -1
                plowerWeight = 1.0
                pupperWeight = 0.0
            else:
                plowerIndex = pIndex[i]
                pupperIndex = pIndex[i]+1
                plowerWeight = (Dist_pGrid[pupperIndex]-perm_vals[i])/(Dist_pGrid[pupperIndex]-Dist_pGrid[plowerIndex])
                pupperWeight = 1.0 - plowerWeight
                
            probGrid[mlowerIndex][plowerIndex] = probGrid[mlowerIndex][plowerIndex] + probs[i]*mlowerWeight*plowerWeight
            probGrid[mlowerIndex][pupperIndex] = probGrid[mlowerIndex][pupperIndex] + probs[i]*mlowerWeight*pupperWeight
            probGrid[mupperIndex][plowerIndex] = probGrid[mupperIndex][plowerIndex] + probs[i]*mupperWeight*plowerWeight
            probGrid[mupperIndex][pupperIndex] = probGrid[mupperIndex][pupperIndex] + probs[i]*mupperWeight*pupperWeight
            
        return probGrid.flatten()
    
    
    def Jump_To_Grid_M(self,m_vals, probs ,Dist_mGrid ):
        '''
        Distributes values onto a predefined grid, maintaining the means.
        ''' 
    
        probGrid = np.zeros(len(Dist_mGrid))
        mIndex = np.digitize(m_vals,Dist_mGrid) - 1
        mIndex[m_vals <= Dist_mGrid[0]] = -1
        mIndex[m_vals >= Dist_mGrid[-1]] = len(Dist_mGrid)-1
        
 
        for i in range(len(m_vals)):
            if mIndex[i]==-1:
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(Dist_mGrid)-1:
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else:
                mlowerIndex = mIndex[i]
                mupperIndex = mIndex[i]+1
                mlowerWeight = (Dist_mGrid[mupperIndex]-m_vals[i])/(Dist_mGrid[mupperIndex]-Dist_mGrid[mlowerIndex])
                mupperWeight = 1.0 - mlowerWeight
                
            probGrid[mlowerIndex] = probGrid[mlowerIndex] + probs[i]*mlowerWeight
            probGrid[mupperIndex] = probGrid[mupperIndex] + probs[i]*mupperWeight
            
        return probGrid.flatten()
    
    def Calc_Ergodic_Dist(self):
        
        '''
        Calculates the egodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is presented as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.
        ''' 
        
        eigen, ergodic_distr = sp.linalg.eigs(self.TranMatrix , k=1 , which='LM')  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        
        self.VecErgDstn = ergodic_distr #distribution as a vector
        self.ErgDstn = ergodic_distr.reshape((len(self.Dist_mGrid),len(self.Dist_pGrid))) # distribution reshaped into len(mgrid) by len(pgrid) array

        
        Edstn = deepcopy(self.ErgDstn)
        for i in range(len(Edstn[0])):
            Edstn[:,i] = Edstn[:,i]*self.Dist_pGrid[i]    
    
        mdstn_weighted = 0
        for i in range (len(ss.ErgDstn[0])):
            mdstn_weighted += Edstn[:,i] # cant be right because the transition matrix doesnt include the reweighted density ??@!!!? MAybe not cuz i don tthink this part actually depends on the density fo that

        self.norm_Dstn = mdstn_weighted
        
    def calc_ergodic_dist_M(self):
        
        eigen, ergodic_distr = sp.linalg.eigs(self.TranMatrix_M , k=1 , which='LM')  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        self.vec_Dstn_M = ergodic_distr
        
        

    def calc_agg_path(self,init_dstn):
        
        '''
        Calculates the path of aggregate consumption and aggregate assets and stores these paths as attributes of self.
        
        
        Parameters
        ----------
        init_dstn: np.array
                Initial distribution of market resources and permanent income

        Returns
        -------
        None
            
        ''' 
        
        AggC =[] # List of aggregate consumption for each period t 
        AggA =[] # List of aggregate assets for each period t 
    
        dstn = init_dstn # Initial distribution set as steady state distribution
    
        T = self.T_cycle
        for i in range(T):
    
            p = self.Dist_pGrid[i]# Permanent income Grid this period
            c = self.cPolGrid[i] # Consumption Policy Grid this period
            a = self.aPolGrid[i] # Asset Policy Grid this period
    
            gridc = np.dot( c.reshape( len(c), 1 ) , p.reshape( 1 , len(p) ) ) #Transform grid from normalized consumption to level of consumption
            C = np.dot( gridc.flatten() , dstn )  # Compute Aggregate Consumption this period
            AggC.append(C)
    
            grida = np.dot( a.reshape( len(a), 1 ) , p.reshape( 1 , len(p) ) ) #Transform grid from normalized assets to level of assets
            A = np.dot( grida.flatten() , dstn ) # Compute Aggregate Assets this period
            AggA.append(A)
    
            dstn = np.dot(self.TranMatrix[i],dstn) # Iterate Distribution forward
    
        #Transform Lists into tractable arrays for plotting
        self.AggC  = np.array(AggC).T[0] 
        self.AggA  = np.array(AggA).T[0]
        
        
    def calc_agg_path_m(self,init_dstn):
        
        
        AggC =[] # List of aggregate consumption for each period t 
        AggA =[] # List of aggregate assets for each period t 
    
        dstn = init_dstn # Initial distribution set as steady state distribution
    
        T = self.T_cycle
        for i in range(T):
    
            c = self.cPolGrid[i] # Consumption Policy Grid this period
            a = self.aPolGrid[i] # Asset Policy Grid this period
    
            C = np.dot( c , dstn )  # Compute Aggregate Consumption this period
            AggC.append(C)
    
            A = np.dot( a, dstn ) # Compute Aggregate Assets this period
            AggA.append(A)
    
            dstn = np.dot(self.TranMatrix_M[i],dstn) # Iterate Distribution forward
    
        #Transform Lists into tractable arrays for plotting
        self.AggC_m  = np.array(AggC).T[0] 
        self.AggA_m = np.array(AggA).T[0]
    



'''
ghost_agent.del_from_time_inv('Rfree') This needs to be changed!
ghost_agent.add_to_time_vary('Rfree')
'''


FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.03**.25,                       # Interest factor on assets
    "DiscFac": 0.96,                   # Intertemporal discount factor
    "LivPrb" : [.99375],                    # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [.05],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.3],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" :  .2,                    # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : .2,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "AgentCount" : 50000,                  # Number of agents of this type
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(1.6)-(.5**2)/2,# Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
     }




ss = JACTran(**FBSDict)
ss.cycles=0
ss.AgentCount = 10000 
ss.T_sim = 2000


ss.solve()
#ss.initialize_sim()
#ss.simulate()


#Consumption = np.mean((ss.state_now['mNrm'] - ss.state_now['aNrm'])*ss.state_now['pLvl'])
#ASSETS = np.mean(ss.state_now['aNrm']*ss.state_now['pLvl'])


ss.Define_Distribution_Grid()


ss.Calc_Transition_Matrix()
ss.Calc_Ergodic_Dist()

vecDstn = ss.VecErgDstn
erg_Dstn = ss.ErgDstn
TranMat = ss.TranMatrix

p = ss.Dist_pGrid
c = ss.cPolGrid
asset = ss.aPolGrid


gridc = np.zeros((len(c),len(p)))
grida = np.zeros((len(asset),len(p)))
gridmu_ss = np.zeros((len(c),len(p)))


for j in range(len(p)):
    gridc[:,j] = p[j]*c
    grida[:,j] = p[j]*asset
    gridmu_ss[:,j] = p[j]* ((p[j]*c) **-ss.CRRA)

Css = np.dot(gridc.flatten(), vecDstn)
AggA = np.dot(grida.flatten(), vecDstn)        
MUss = np.dot(gridmu_ss.flatten(), vecDstn) 

#aa = gridc.flatten()

#aa = aa.reshape(len(c),len(p))

#DDstn = vecDstn.reshape(len(c),len(p))

#print('MRS =' + str(MRS))
#print('what it needs to be:' + str(AMRS))
print('Assets =' + str(AggA))
#print('simulated assets = ' +str(ASSETS))
print('consumption =' + str(Css))
#print('simulated Consumption = ' +str(Consumption))


ss.Calc_Transition_Matrix_M()
ss.calc_ergodic_dist_M()



C_ntrl = np.dot(c,ss.vec_Dstn_M)
A_ntrl =np.dot(asset,ss.vec_Dstn_M)

print('difference in assets =' + str(A_ntrl-AggA))
print('difference in consumption =' + str(C_ntrl-Css))



#------------------------------------------------------------------

          
mdstn = []
                
for i in range (len(ss.ErgDstn[:,0])):
    mdstn.append(np.sum(ss.ErgDstn[i]))
    
mdstn = np.array(mdstn) # marginal distribution of normalized market resources


mdstn2 = 0                
for i in range (len(ss.ErgDstn[0])):
    mdstn2 +=ss.ErgDstn[:,i]
    
#mdstn2 = np.array(mdstn2) # marginal distribution of normalized market resources


Edstn = deepcopy(ss.ErgDstn)
for i in range(len(Edstn[0])):
    Edstn[:,i] = Edstn[:,i]*ss.Dist_pGrid[i]    
    
mdstn_weighted = 0
for i in range (len(ss.ErgDstn[0])):
    mdstn_weighted += Edstn[:,i] # cant be right because the transition matrix doesnt include the reweighted density ??@!!!? MAybe not cuz i don tthink this part actually depends on the density fo that


#permanent income weighted distribution

C = np.dot(ss.cPolGrid, mdstn_weighted)
A = np.dot(ss.aPolGrid,mdstn_weighted)
print(A) #This Works!
print(C) #This Works too!!












#------------------------------------------------------------------------------




params = deepcopy(FBSDict)
params['T_cycle']= 30
params['LivPrb']= params['T_cycle']*[ss.LivPrb[0]]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[ss.PermShkStd[0]]
params['TranShkStd']= params['T_cycle']*[ss.TranShkStd[0]]
params['Rfree'] = params['T_cycle']*[ss.Rfree]



example = JACTran(**params)
example.cycles = 1

dx=.01
example.del_from_time_inv('Rfree')
example.add_to_time_vary('Rfree')
example.IncShkDstn = params['T_cycle']*ss.IncShkDstn
example.IncShkDstn_ntrl_msr = params['T_cycle']*ss.IncShkDstn_ntrl_msr

example.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)



i = 15
example.Rfree = (i)*[ss.Rfree] + [ss.Rfree + dx] + (params['T_cycle'] - i - 1 )*[ss.Rfree]
#example.PermShkStd = (i)*ss.PermShkStd + [ss.PermShkStd[0] + dx] + (params['T_cycle'] - i )*ss.PermShkStd 



#TranShkDstnP     = MeanOneLogNormal(ss.TranShkStd[0],123).approx(ss.TranShkCount)
#PermShkDstnP     = MeanOneLogNormal(ss.PermShkStd[0] + dx ,123).approx(ss.PermShkCount)
#IncShkDstnP= [combine_indep_dstns(PermShkDstnP,TranShkDstnP)]
        
#example.IncShkDstn = i*ss.IncShkDstn + IncShkDstnP + (params['T_cycle'] - i - 1)* ss.IncShkDstn
#example.update_income_process()
    
    
example.solve()
example.Define_Distribution_Grid()


start = time.time()
example.Calc_Transition_Matrix()
#example.Calc_Ergodic_Dist()
example.calc_agg_path(vecDstn)
print('first time =' + str(time.time()-start))

start2 = time.time()
example.Calc_Transition_Matrix_M()
example.calc_agg_path_m(ss.vec_Dstn_M)
print('second time =' + str(time.time()-start2))


'''
Distributions = []


AggC_List =[]
AggA_List =[]
T=params['T_cycle']

dstn = vecDstn

for i in range(T):

    p = example.Dist_pGrid[i]
    c = example.cPolGrid[i]
    a = example.aPolGrid[i]

    gridc = np.dot( c.reshape( len(c), 1 ) , p.reshape( 1 , len(p) ) )
    C = np.dot( gridc.flatten() , dstn )
    AggC_List.append(C)
    
    
    grida = np.dot( a.reshape( len(a), 1 ) , p.reshape( 1 , len(p) ) )
    A = np.dot( grida.flatten() , dstn )
    AggA_List.append(A)
    
    dstn = np.dot(example.TranMatrix[i],dstn)

AggC_List  = np.array(AggC_List).T[0]
AggA_List  = np.array(AggA_List).T[0]


plt.plot(AggC_List)
plt.plot(np.ones( len(AggC_List) )* 1.00676409, 'k')
#plt.ylim([1.004,1.01])
plt.show()

plt.plot(AggA_List)
plt.plot(np.ones( len(AggA_List) )* AggA, 'k')
#plt.ylim([1.004,1.01])
plt.show()
    
'''



plt.plot(example.AggC)
plt.plot(np.ones( len(example.AggA) )* Css, 'k')
#plt.ylim([1.004,1.01])
plt.show()

plt.plot(example.AggA)
plt.plot(np.ones( len(example.AggA) )* AggA, 'k')
#plt.ylim([1.004,1.01])
plt.show()


plt.plot(example.AggC_m, 'g')
plt.plot(example.AggC)
plt.plot(np.ones( len(example.AggC_m) )* C_ntrl, 'r')
#plt.ylim([1.004,1.01])
plt.show()

axtrashift = np.delete(ss.aXtraGrid,-1)
axtrashift = np.insert(axtrashift, 0,1.00000000e-04)
dist_betw_pts = ss.aXtraGrid - axtrashift
dist_betw_pts_half = dist_betw_pts/2

newAgrid = axtrashift + dist_betw_pts_half
densergrid = np.concatenate((ss.aXtraGrid,newAgrid))

#plt.hist(ss.aXtraGrid,l)


l =np.linspace(0,20,num=100)
plt.hist(densergrid,l)
aXtraGrid = ss.aXtraGrid

for i in range(0):
    axtrashift = np.delete(aXtraGrid,-1) 
    axtrashift = np.insert(axtrashift, 0,1.00000000e-04)
    dist_betw_pts = aXtraGrid - axtrashift
    dist_betw_pts_half = dist_betw_pts/2
    newAgrid = axtrashift + dist_betw_pts_half
    aXtraGrid = np.concatenate((aXtraGrid,newAgrid))
    aXtraGrid = np.sort(aXtraGrid)
    
print(aXtraGrid - ss.aXtraGrid)
plt.hist(aXtraGrid,l)

a = np.zeros(len(aXtraGrid))

plt.plot(aXtraGrid,a, '.')

plt.plot(ss.aXtraGrid, np.zeros(len(ss.aXtraGrid)), '.')


'''
                
                axtrashift = np.delete(self.aXtraGrid,-1) 
                axtrashift = np.insert(axtrashift, 0,1.00000000e-04)
                dist_betw_pts = self.aXtraGrid - axtrashift
                dist_betw_pts_half = dist_betw_pts/2
                newAgrid = axtrashift + dist_betw_pts_half
                densergrid = np.concatenate((self.aXtraGrid,newAgrid))
                densergrid = np.sort(densergrid)
                '''

ntrl = deepcopy(ss.PermShkDstn)
ntrl[0].pmf = ntrl[0].X*ntrl[0].pmf




