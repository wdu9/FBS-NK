# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 20:33:18 2021

@author: wdu
"""


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
import time 
from scipy.io import savemat


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
    
    
    time_inv_ = IndShockConsumerType.time_inv_  + [
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
        

        TranShkDstn = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstn.pmf  = np.insert(TranShkDstn.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)   
        TranShkDstn.X  = np.insert(TranShkDstn.X*(((1.0-self.tax_rate)*self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstn     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstn = [combine_indep_dstns2(PermShkDstn,TranShkDstn)]
        self.TranShkDstn = [TranShkDstn]
        self.PermShkDstn = [PermShkDstn]
        self.add_to_time_vary('IncShkDstn')
        
        
        PermShk_ntrl_msr = deepcopy(PermShkDstn)
        PermShk_ntrl_msr.pmf = PermShk_ntrl_msr.X*PermShk_ntrl_msr.pmf
        self.IncShkDstn_ntrl_msr = [combine_indep_dstns(PermShk_ntrl_msr,TranShkDstn)]
        
        PermShk_ntrl_msr_1 = deepcopy(PermShkDstn)
        PermShk_ntrl_msr_1.pmf = PermShk_ntrl_msr_1.X**(-1) * PermShk_ntrl_msr_1.pmf
        self.IncShkDstn_ntrl_msr_1 = [combine_indep_dstns(PermShk_ntrl_msr_1,TranShkDstn)]
      

        TranShkDstnW     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnW.pmf  = np.insert(TranShkDstnW.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnW.X  = np.insert(TranShkDstnW.X*(((1.0-self.tax_rate)*self.N*(self.wage + self.dx))/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnW     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnW = [combine_indep_dstns2(PermShkDstnW,TranShkDstnW)]
        
        self.IncShkDstnW_ntrl_msr = [combine_indep_dstns2(PermShk_ntrl_msr,TranShkDstnW)]
        self.IncShkDstnW_ntrl_msr_1 = [combine_indep_dstns2(PermShk_ntrl_msr_1,TranShkDstnW)]

        self.TranShkDstnW = [TranShkDstnW]
        self.PermShkDstnW = [PermShkDstnW]
        self.add_to_time_vary('IncShkDstnW')
        
        
     
        TranShkDstnN     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnN.pmf  = np.insert(TranShkDstnN.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnN.X  = np.insert(TranShkDstnN.X*(((1.0-self.tax_rate)*(self.N + self.dx)*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnN     = MeanOneLogNormal(self.PermShkStd[0],123).approx(self.PermShkCount)
        self.IncShkDstnN= [combine_indep_dstns2(PermShkDstnN,TranShkDstnN)]
        
        self.IncShkDstnN_ntrl_msr = [combine_indep_dstns2(PermShk_ntrl_msr,TranShkDstnN)]

        self.TranShkDstnN = [TranShkDstnN]
        self.PermShkDstnN = [PermShkDstnN]
        self.add_to_time_vary('IncShkDstnN')
        
        
        TranShkDstnP     = MeanOneLogNormal(self.TranShkStd[0],123).approx(self.TranShkCount)
        TranShkDstnP.pmf  = np.insert(TranShkDstnP.pmf*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstnP.X  = np.insert(TranShkDstnP.X*(((1.0- self.tax_rate) *self.N*self.wage)/(1-self.UnempPrb)),0,self.IncUnemp)
        PermShkDstnP     = MeanOneLogNormal(self.PermShkStd[0] + self.dx ,123).approx(self.PermShkCount)
        self.IncShkDstnP= [combine_indep_dstns2(PermShkDstnP,TranShkDstnP)]
        
        PermShkP_ntrl_msr = deepcopy(PermShkDstnP)
        PermShkP_ntrl_msr.pmf = PermShkP_ntrl_msr.X*PermShkP_ntrl_msr.pmf
        self.IncShkDstnP_ntrl_msr = [combine_indep_dstns2(PermShkP_ntrl_msr,TranShkDstnN)]

        self.TranShkDstnP = [TranShkDstnP]
        self.PermShkDstnP = [PermShkDstnP]
        self.add_to_time_vary('IncShkDstnP')
        
        
        
        
        
        
    
     
    def define_distribution_grid(self, dist_mGrid=None, dist_pGrid=None, m_density = 0, num_pointsM = 48,  num_pointsP = 50, max_p_fac = 20.0):
        
        '''
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and permanent income may be prespecified 
        as dist_mGrid and dist_pGrid, respectively. If not then default grid is computed based off given parameters.
        
        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources
            
        dist_pGrid : np.array
                Prespecified grid for distribution over permanent income. 
            
        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.
            
        num_pointsM: float
                Number of gridpoints for market resources grid.
        
        num_pointsP: float
                 Number of gridpoints for permanent income. 
                 This grid will be exponentiated by the function make_grid_exp_mult.
                
        max_p_fac : float
                Factor that scales the maximum value of permanent income grid. 
                Larger values increases the maximum value of permanent income grid.
        
        Returns
        -------
        None
        '''  
 
        if self.cycles == 0:
            if dist_mGrid == None:    
                aXtra_Grid = make_grid_exp_mult(
                        ming=self.aXtraMin, maxg=self.aXtraMax, ng = num_pointsM, timestonest = 3) #Generate Market resources grid given density and number of points
                
                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid,-1) 
                    axtra_shifted = np.insert(axtra_shifted, 0,1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts/2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid,new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)
                    
                self.dist_mGrid =  aXtra_Grid
            else:
                self.dist_mGrid = dist_mGrid #If grid of market resources prespecified then use as mgrid
                
            if dist_pGrid == None:
                num_points = num_pointsP #Number of permanent income gridpoints
                #Dist_pGrid is taken to cover most of the ergodic distribution
                p_variance = self.PermShkStd[0]**2 #set variance of permanent income shocks
                max_p = max_p_fac*(p_variance/(1-self.LivPrb[0]))**0.5 #Maximum Permanent income value
                one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2)
                self.dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid) #Compute permanent income grid
            else:
                self.dist_pGrid = dist_pGrid #If grid of permanent income prespecified then use it as pgrid
                
        elif self.cycles > 1:
            print('define_distribution_grid requires cycles = 0 or cycles = 1')
        
        elif self.T_cycle != 0:
            
            if dist_mGrid == None:
                aXtra_Grid = make_grid_exp_mult(
                        ming=self.aXtraMin, maxg=self.aXtraMax, ng = num_pointsM, timestonest = 3) #Generate Market resources grid given density and number of points
                
                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid,-1) 
                    axtra_shifted = np.insert(axtra_shifted, 0,1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts/2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid,new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)
                    
                self.dist_mGrid =  aXtra_Grid
                
            else:
                self.dist_mGrid = dist_mGrid #If grid of market resources prespecified then use as mgrid
                    
            if dist_pGrid == None:
                
                self.dist_pGrid = [] #list of grids of permanent income    
                
                for i in range(self.T_cycle):
                    
                    num_points = 50
                    #Dist_pGrid is taken to cover most of the ergodic distribution
                    p_variance = self.PermShkStd[i]**2 # set variance of permanent income shocks this period
                    max_p = 20.0*(p_variance/(1-self.LivPrb[i]))**0.5 # Consider probability of staying alive this period
                    one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2) 
                    
                    dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid) # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    self.dist_pGrid.append(dist_pGrid)

            else:
                self.dist_pGrid = dist_pGrid #If grid of permanent income prespecified then use as pgrid
                
                
    def calc_transition_matrix(self, shk_dstn = None):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem. 
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.
        
        
        Parameters
        ----------
            shk_dstn: list 
                list of income shock distributions

        Returns
        -------
        None
        
        ''' 
        
        
        if self.cycles == 0: 
            
            if shk_dstn == None:
                shk_dstn = self.IncShkDstn
            
            dist_mGrid = self.dist_mGrid #Grid of market resources
            dist_pGrid = self.dist_pGrid #Grid of permanent incomes
            aNext = dist_mGrid - self.solution[0].cFunc(dist_mGrid)  #assets next period
            
            self.aPol_Grid = aNext # Steady State Asset Policy Grid
            self.cPol_Grid = self.solution[0].cFunc(dist_mGrid) #Steady State Consumption Policy Grid
            
            # Obtain shock values and shock probabilities from income distribution
            bNext = self.Rfree*aNext # Bank Balances next period (Interest rate * assets)
            shk_prbs = shk_dstn[0].pmf  # Probability of shocks 
            tran_shks = shk_dstn[0].X[1] # Transitory shocks
            perm_shks = shk_dstn[0].X[0] # Permanent shocks
            LivPrb = self.LivPrb[0] # Update probability of staying alive
            
            #New borns have this distribution (assumes start with no assets and permanent income=1)
            NewBornDist = self.jump_to_grid(tran_shks,np.ones_like(tran_shks),shk_prbs,dist_mGrid,dist_pGrid)
            
            # Generate Transition Matrix
            TranMatrix = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid)))
            for i in range(len(dist_mGrid)):
                for j in range(len(dist_pGrid)):
                    mNext_ij = bNext[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                    pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                    TranMatrix[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs,dist_mGrid,dist_pGrid) + (1.0-LivPrb)*NewBornDist
            self.tran_matrix = TranMatrix
            
            
        elif self.cycles > 1:
            print('calc_transition_matrix requires cycles = 0 or cycles = 1')
            
        elif self.T_cycle!= 0:
            
            if shk_dstn == None:
                shk_dstn = self.IncShkDstn
            
            self.cPol_Grid = [] # List of consumption policy grids for each period in T_cycle
            self.aPol_Grid = [] # List of asset policy grids for each period in T_cycle
            self.tran_matrix = [] # List of transition matrices
            
            dist_mGrid =  self.dist_mGrid
            
            for k in range(self.T_cycle):
                           
                if type(self.dist_pGrid) == list:
                    dist_pGrid = self.dist_pGrid[k] #Permanent income grid this period
                else:
                    dist_pGrid = self.dist_pGrid #If here then use prespecified permanent income grid
                
                Cnow = self.solution[k].cFunc(dist_mGrid) #Consumption policy grid in period k
                self.cPol_Grid.append(Cnow) #Add to list

                aNext = dist_mGrid - Cnow # Asset policy grid in period k
                self.aPol_Grid.append(aNext) # Add to list
                
                
                if type(self.Rfree)==list:
                    bNext = self.Rfree[k]*aNext
                else:
                    bNext = self.Rfree*aNext
                    
                #Obtain shocks and shock probabilities from income distribution this period
                shk_prbs = shk_dstn[k].pmf  #Probability of shocks this period
                tran_shks = shk_dstn[k].X[1] #Transitory shocks this period
                perm_shks = shk_dstn[k].X[0] #Permanent shocks this period
                LivPrb = self.LivPrb[k] # Update probability of staying alive this period
                
                
                
                if len(dist_pGrid) == 1: 
            
                    #New borns have this distribution (assumes start with no assets and permanent income=1)
                    NewBornDist = self.jump_to_grid_fast(tran_shks,shk_prbs,dist_mGrid)
                    
                    # Generate Transition Matrix this period
                    TranMatrix_M = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                    for i in range(len(dist_mGrid)):
                            mNext_ij = bNext[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                            TranMatrix_M[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist 
                    TranMatrix_M = TranMatrix_M
                    self.tran_matrix.append(TranMatrix_M)
                    
                else:
                    
                    NewBornDist = self.jump_to_grid(tran_shks,np.ones_like(tran_shks),shk_prbs,dist_mGrid,dist_pGrid)

                    # Generate Transition Matrix this period
                    TranMatrix = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid))) 
                    for i in range(len(dist_mGrid)):
                        for j in range(len(dist_pGrid)):
                            mNext_ij = bNext[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                            pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                            TranMatrix[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs, dist_mGrid, dist_pGrid) + (1.0-LivPrb)*NewBornDist #generate transition probabilities
                    TranMatrix = TranMatrix #columns represent the current state while rows represent the next state
                    #the 4th row , 6th column entry represents the probability of transitioning from the 6th element of the combined perm and m grid (grid of market resources multiplied by grid of perm income) to the 4th element of the combined perm and m grid
                    self.tran_matrix.append(TranMatrix)         
                


                
    def jump_to_grid(self, m_vals, perm_vals, probs, dist_mGrid, dist_pGrid ):
        
        '''
        Distributes values onto a predefined grid, maintaining the means. m_vals and perm_vals are realizations of market resources and permanent income while 
        dist_mGrid and dist_pGrid are the predefined grids of market resources and permanent income, respectively. That is, m_vals and perm_vals do not necesarily lie on their 
        respective grids. Returns probabilities of each gridpoint on the combined grid of market resources and permanent income.
        
        
        Parameters
        ----------
        m_vals: np.array
                Market resource values 
        
        perm_vals: np.array
                Permanent income values 
        
        probs: np.array
                Shock probabilities associated with combinations of m_vals and perm_vals. 
                Can be thought of as the probability mass function  of (m_vals, perm_vals).
        
        dist_mGrid : np.array
                Grid over normalized market resources
            
        dist_pGrid : np.array
                Grid over permanent income 

        Returns
        -------
        probGrid.flatten(): np.array
                 Probabilities of each gridpoint on the combined grid of market resources and permanent income
        '''
    
        probGrid = np.zeros((len(dist_mGrid),len(dist_pGrid)))
        mIndex = np.digitize(m_vals,dist_mGrid) - 1 # Array indicating in which bin each values of m_vals lies in relative to dist_mGrid. Bins lie between between point of Dist_mGrid. 
        #For instance, if mval lies between dist_mGrid[4] and dist_mGrid[5] it is in bin 4 (would be 5 if 1 was not subtracted in the previous line). 
        mIndex[m_vals <= dist_mGrid[0]] = -1 # if the value is less than the smallest value on dist_mGrid assign it an index of -1
        mIndex[m_vals >= dist_mGrid[-1]] = len(dist_mGrid)-1 # if value if greater than largest value on dist_mGrid assign it an index of the length of the grid minus 1
        
        #the following three lines hold the same intuition as above
        pIndex = np.digitize(perm_vals,dist_pGrid) - 1
        pIndex[perm_vals <= dist_pGrid[0]] = -1
        pIndex[perm_vals >= dist_pGrid[-1]] = len(dist_pGrid)-1
        
        for i in range(len(m_vals)):
            if mIndex[i]==-1: # if mval is below smallest gridpoint, then assign it a weight of 1.0 for lower weight. 
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(dist_mGrid)-1: # if mval is greater than maximum gridpoint, then assign the following weights
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else: # Standard case where mval does not lie past any extremes
            #identify which two points on the grid the mval is inbetween
                mlowerIndex = mIndex[i] 
                mupperIndex = mIndex[i]+1
            #Assign weight to the indices that bound the m_vals point. Intuitively, an mval perfectly between two points on the mgrid will assign a weight of .5 to the gridpoint above and below
                mlowerWeight = (dist_mGrid[mupperIndex]-m_vals[i])/(dist_mGrid[mupperIndex]-dist_mGrid[mlowerIndex]) #Metric to determine weight of gridpoint/index below. Intuitively, mvals that are close to gridpoint/index above are assigned a smaller mlowerweight.
                mupperWeight = 1.0 - mlowerWeight # weight of gridpoint/ index above
                
            #Same logic as above except the weights here concern the permanent income grid
            if pIndex[i]==-1: 
                plowerIndex = 0
                pupperIndex = 0
                plowerWeight = 1.0
                pupperWeight = 0.0
            elif pIndex[i]==len(dist_pGrid)-1:
                plowerIndex = -1
                pupperIndex = -1
                plowerWeight = 1.0
                pupperWeight = 0.0
            else:
                plowerIndex = pIndex[i]
                pupperIndex = pIndex[i]+1
                plowerWeight = (dist_pGrid[pupperIndex]-perm_vals[i])/(dist_pGrid[pupperIndex]-dist_pGrid[plowerIndex])
                pupperWeight = 1.0 - plowerWeight
                
            # Compute probabilities of each gridpoint on the combined market resources and permanent income grid by looping through each point on the combined market resources and permanent income grid, 
            # assigning probabilities to each gridpoint based off the probabilities of the surrounding mvals and pvals and their respective weights placed on the gridpoint.
            # Note* probs[i] is the probability of mval AND pval occurring
            probGrid[mlowerIndex][plowerIndex] = probGrid[mlowerIndex][plowerIndex] + probs[i]*mlowerWeight*plowerWeight # probability of gridpoint below mval and pval 
            probGrid[mlowerIndex][pupperIndex] = probGrid[mlowerIndex][pupperIndex] + probs[i]*mlowerWeight*pupperWeight # probability of gridpoint below mval and above pval
            probGrid[mupperIndex][plowerIndex] = probGrid[mupperIndex][plowerIndex] + probs[i]*mupperWeight*plowerWeight # probability of gridpoint above mval and below pval
            probGrid[mupperIndex][pupperIndex] = probGrid[mupperIndex][pupperIndex] + probs[i]*mupperWeight*pupperWeight # probability of gridpoint above mval and above pval
            

        return probGrid.flatten()

    def jump_to_grid_fast(self,m_vals, probs ,Dist_mGrid ):
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
    
        
    def calc_ergodic_dist(self, transition_matrix = None):
        
        '''
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.
        
        Parameters
        ----------
        transition_matrix: List 
                    transition matrix whose ergordic distribution is to be solved

        Returns
        -------
        None
        '''
        
        if transition_matrix == None:
            transition_matrix = [self.tran_matrix]
        
        
        eigen, ergodic_distr = sp.linalg.eigs(transition_matrix[0] , k=1 , which='LM')  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        
        self.vec_erg_dstn = ergodic_distr #distribution as a vector
        self.erg_dstn = ergodic_distr.reshape((len(self.dist_mGrid),len(self.dist_pGrid))) # distribution reshaped into len(mgrid) by len(pgrid) array
        


    def calc_agg_path(self, init_dstn, cPolGrid_list = None, aPolGrid_list = None, tran_matrix_list = None, dist_pGrid_list =None):
        
        '''
        Calculates the paths of aggregate consumption and aggregate assets storing both as attributes of self. The consumption and asset policies along with the 
        the transition matrices each period may be specified as lists. 
        
        Parameters
        ----------
        init_dstn: np.array
                Initial distribution of market resources and permanent income
                
        cPolGrid_list: list
                list of consumption policy grids
        
        aPolGrid_list: list
                list of asset policy grids
            
        TranMatrix_list: list
                list of transition matrices
                
        Returns
        -------
        None
            
        ''' 
        
        if cPolGrid_list == None:
            cPolGrid_list = self.cPol_Grid
            
        if aPolGrid_list == None:
            aPolGrid_list = self.aPol_Grid
            
        if tran_matrix_list == None:
            tran_matrix_list = self.tran_matrix
            
        if dist_pGrid_list == None:
            dist_pGrid_list = self.dist_pGrid
                        
        AggC =[] # List of aggregate consumption for each period t 
        AggA =[] # List of aggregate assets for each period t 
    
        dstn = init_dstn # Initial distribution set as steady state distribution
    
        T = len(cPolGrid_list)
        for i in range(T):
            
            p = dist_pGrid_list[i]
            c = cPolGrid_list[i] # Consumption Policy Grid this period
            a = aPolGrid_list[i] # Asset Policy Grid this period
            
            if len(p) == 1:
                
                C = np.dot( c , dstn )  # Compute Aggregate Consumption this period
                AggC.append(C)
        
                A = np.dot( a, dstn ) # Compute Aggregate Assets this period
                AggA.append(A)
                
            else:
                
                gridc = np.dot( c.reshape( len(c), 1 ) , p.reshape( 1 , len(p) ) ) #Transform grid from normalized consumption to level of consumption
                C = np.dot( gridc.flatten() , dstn )  # Compute Aggregate Consumption this period
                AggC.append(C)
        
                grida = np.dot( a.reshape( len(a), 1 ) , p.reshape( 1 , len(p) ) ) #Transform grid from normalized assets to level of assets
                A = np.dot( grida.flatten() , dstn ) # Compute Aggregate Assets this period
                AggA.append(A)
                                
            dstn = np.dot(tran_matrix_list[i],dstn) # Iterate Distribution forward
    
        #Transform Lists into tractable arrays
        self.AggC  = np.array(AggC).T[0] 
        self.AggA  = np.array(AggA).T[0]


    def calc_MU(self, init_dstn, cPolGrid_list = None, aPolGrid_list = None, tran_matrix_list = None, dist_pGrid_list =None):
        
        '''
        Calculates the paths of aggregate consumption and aggregate assets storing both as attributes of self. The consumption and asset policies along with the 
        the transition matrices each period may be specified as lists. 
        
        Parameters
        ----------
        init_dstn: np.array
                Initial distribution of market resources and permanent income
                
        cPolGrid_list: list
                list of consumption policy grids
        
        aPolGrid_list: list
                list of asset policy grids
            
        TranMatrix_list: list
                list of transition matrices
                
        Returns
        -------
        None
            
        ''' 
        
        if cPolGrid_list == None:
            cPolGrid_list = self.cPol_Grid
            
        if aPolGrid_list == None:
            aPolGrid_list = self.aPol_Grid
            
        if tran_matrix_list == None:
            tran_matrix_list = self.tran_matrix
            
        if dist_pGrid_list == None:
            dist_pGrid_list = self.dist_pGrid
                        
        MU_list =[] # List of aggregate consumption for each period t 
    
        dstn = init_dstn # Initial distribution set as steady state distribution
    
        T = len(cPolGrid_list)
        for i in range(T):
            
            p = dist_pGrid_list[i]
            c = cPolGrid_list[i] # Consumption Policy Grid this period
            a = aPolGrid_list[i] # Asset Policy Grid this period
            
            if len(p) == 1:
                
                MU = np.dot( c**(-self.CRRA) , dstn )  # Compute Aggregate Consumption this period
                MU_list.append(MU)
        
  
            dstn = np.dot(tran_matrix_list[i],dstn) # Iterate Distribution forward
    
        #Transform Lists into tractable arrays
        self.MU  = np.array(MU_list).T[0] 



FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.05**.25,                 # Interest factor on assets
    "DiscFac": 0.977, #.96,            # Intertemporal discount factor
    "LivPrb" : [.99375],                # Survival probability
    "PermGroFac" :[1.00],               # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [.05],#[(.005*4/11)**.5], #[(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],                   # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.07, #.08                # Probability of unemployment while working
    "IncUnemp" :  .05,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : .1,                      # Flat income tax rate (legacy parameter, will be removed in future)

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
    "AgentCount" : 100000,                 # Number of agents of this type
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(1.6)-(.5**2)/2,# Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
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
     "SSPmu" :  1.01,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : .1,                               # Net Bond Supply
     "G" : .105,
     "DisULabor": 0.8823685356415617,
     "InvFrisch": 2 ,
     "s" : 1
     }





G=.105
t=.1 #0.16563445378151262
Inc = .05
mho=.07
r = (1.05)**.25 - 1 
B=.1 

w = (1/1.01)
N = (Inc*mho + G  + (1 - (1/(1+r)) ) *B) / (w*t) 
q = ((1-w)*N)/r

A = ( B/(1+r) ) + q

#('output =' +str(N))
#print(q)
#print('Target Consumption =' +str(N-G))
#print('Target Assets =' +str(A))

target = A

ss = JACTran(**FBSDict)
ss.cycles=0
ss.dx=0
ss.T_sim = 1000

norm = ((1-ss.UnempPrb)/((ss.wage) * ss.N * (1 - ss.tax_rate)))

tolerance = .01

completed_loops=0

go = True

DiscFac = ss.DiscFac

while go:
    
    ss.DiscFac = DiscFac
    
    ss.solve()
    #ss.initialize_sim()
    #ss.simulate()
    
    #Consumption = np.mean((ss.state_now['mNrm'] - ss.state_now['aNrm'])*ss.state_now['pLvl'])
    #ASSETS = np.mean(ss.state_now['aNrm']*ss.state_now['pLvl'])

    
    ss.define_distribution_grid(dist_pGrid = np.array([1]))
    ss.calc_transition_matrix(ss.IncShkDstn_ntrl_msr)
    ss.calc_ergodic_dist([ss.tran_matrix])

    SS_dstn = ss.vec_erg_dstn
    c = ss.cPol_Grid
    asset = ss.aPol_Grid
    
    Css =  np.dot(c,SS_dstn)
    AggA =np.dot(asset,SS_dstn)
    
    
    ss.define_distribution_grid(dist_pGrid = np.array([1]))
    ss.calc_transition_matrix(ss.IncShkDstn_ntrl_msr_1)
    ss.calc_ergodic_dist([ss.tran_matrix])
    
    MU_dstn = ss.vec_erg_dstn
    
    MU = np.dot(c**(-ss.CRRA), MU_dstn)
    
    dif = AggA - target
    
    
    if dif[0] > 0 :
        
       DiscFac = DiscFac - dif[0]/200
        
    elif dif[0] < 0: 
        DiscFac = DiscFac- dif[0]/200
        
    else:
        break
    
    
    print('MU =' + str(MU))

    print('Assets =' + str(AggA))
    #print('simulated assets = ' +str(ASSETS))
    print('Target Assets =' +str(A))


    print('consumption =' + str(Css))
    #print('simulated Consumption = ' +str(Consumption))
    print('Target Consumption =' +str(N-G))


    print('DiscFac =' + str(DiscFac))


    distance = abs(dif[0]) 
    
    completed_loops += 1
    
    print('Completed loops:' + str(completed_loops))
    
    go = distance >= tolerance and completed_loops < 1
        
print("Done Computing Steady State")






params = deepcopy(FBSDict)
params['T_cycle'] = 200
params['LivPrb']= params['T_cycle']*[ss.LivPrb[0]]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[ss.PermShkStd[0]]
params['TranShkStd']= params['T_cycle']*[ss.TranShkStd[0]]
params['Rfree'] = params['T_cycle']*[ss.Rfree]




ghost = JACTran(**params)
ghost.pseudo_terminal = False
ghost.cycles = 1
ghost.IncShkDstn_ntrl_msr = params['T_cycle']*ss.IncShkDstn_ntrl_msr
ghost.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)

ghost.jac = False
ghost.jacW = False
ghost.jacN = False
ghost.jacPerm = False

ghost.IncShkDstn_ntrl_msr_1 = params['T_cycle']*ss.IncShkDstn_ntrl_msr_1
ghost.del_from_time_inv('Rfree')
ghost.add_to_time_vary('Rfree')
ghost.IncShkDstn = params['T_cycle']*ss.IncShkDstn


ghost.solve()

ghost.define_distribution_grid(dist_pGrid = ghost.T_cycle*[np.array([1])])
ghost.calc_transition_matrix(ghost.IncShkDstn_ntrl_msr_1)
ghost.calc_MU(init_dstn = MU_dstn)

MU_ghost = ghost.MU

plt.plot(MU_ghost)









#Jacobian

example = JACTran(**params)
example.pseudo_terminal = False
example.cycles = 1
example.IncShkDstn_ntrl_msr = params['T_cycle']*ss.IncShkDstn_ntrl_msr
example.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)


example.jac = False
example.jacW = True
example.jacN = False
example.jacPerm = False


if example.jac == True:
    example.dx = .0001
    example.del_from_time_inv('Rfree')
    example.add_to_time_vary('Rfree')
    example.IncShkDstn = params['T_cycle']*ss.IncShkDstn
    example.IncShkDstn_ntrl_msr = params['T_cycle']*ss.IncShkDstn_ntrl_msr
    example.IncShkDstn_ntrl_msr_1 = params['T_cycle']*ss.IncShkDstn_ntrl_msr_1


    
if example.jacW==True or example.jacN == True or example.jacPerm ==True:
    example.dx = .0001 
    example.Rfree = ss.Rfree
    example.update_income_process()
    



CHist=[]
AHist=[]
MUHist=[]
T=params['T_cycle']

start1 = time.time()

for q in range(T):
#testset = [0,15]

#for q in testset:
    
    if example.jac == True:
        example.Rfree = q*[ss.Rfree] + [ss.Rfree + example.dx] + (params['T_cycle'] - q )*[ss.Rfree]
    
    if example.jacW == True:
        example.IncShkDstn = q*ss.IncShkDstn + example.IncShkDstnW + (params['T_cycle'] - q )* ss.IncShkDstn
        example.IncShkDstn_ntrl_msr = q*ss.IncShkDstn_ntrl_msr + example.IncShkDstnW_ntrl_msr + (params['T_cycle'] - q )* ss.IncShkDstn_ntrl_msr
        example.IncShkDstn_ntrl_msr_1 = q*ss.IncShkDstn_ntrl_msr_1 + example.IncShkDstnW_ntrl_msr_1 + (params['T_cycle'] - q )* ss.IncShkDstn_ntrl_msr_1

    if example.jacN == True:
        example.IncShkDstn = q*ss.IncShkDstn + example.IncShkDstnN + (params['T_cycle'] - q )* ss.IncShkDstn
        example.IncShkDstn_ntrl_msr =q*ss.IncShkDstn_ntrl_msr + example.IncShkDstnN_ntrl_msr + (params['T_cycle'] - q )* ss.IncShkDstn_ntrl_msr
        
    if example.jacPerm == True:
        example.IncShkDstn = q*ss.IncShkDstn + example.IncShkDstnP + (params['T_cycle'] - q )* ss.IncShkDstn
        example.IncShkDstn_ntrl_msr =q*ss.IncShkDstn_ntrl_msr + example.IncShkDstnP_ntrl_msr + (params['T_cycle'] - q )* ss.IncShkDstn_ntrl_msr

    
    example.solve()
    
    
    start = time.time()

    example.define_distribution_grid(dist_pGrid = example.T_cycle*[np.array([1])])
    example.calc_transition_matrix(example.IncShkDstn_ntrl_msr)
    example.calc_agg_path(init_dstn = SS_dstn)
    
    print('seconds past : ' + str(time.time()-start))
    
    CHist.append(example.AggC)
    AHist.append(example.AggA)
    
    
    example.define_distribution_grid(dist_pGrid = example.T_cycle*[np.array([1])])
    example.calc_transition_matrix(example.IncShkDstn_ntrl_msr_1)
    example.calc_MU(init_dstn = MU_dstn)
    
    MUHist.append(example.MU)
   
    print(q)

print('seconds past : ' + str(time.time()-start1))

    
plt.plot((CHist[0]- Css[0])/example.dx, label = '0' )
plt.plot((CHist[20]- Css[0])/example.dx, label = '20' )
plt.plot((CHist[30]- Css[0])/example.dx, label = '30' )
plt.plot(np.zeros(len(CHist[0])), color = 'k')
plt.legend()
plt.show()


plt.plot((AHist[0]- AggA[0])/example.dx, label = '0' )
plt.plot((AHist[1]- AggA[0])/example.dx, label = '20' )
plt.plot((AHist[30]- AggA[0])/example.dx, label = '20' )
#plt.plot((AHist[50]- AggA[0])/example.dx, label = '20' )

plt.plot(np.zeros(len(CHist[0])), color = 'k')
plt.legend()
plt.show()



plt.plot((MUHist[0]- MU_ghost)/example.dx, label = '0' )
plt.plot((MUHist[30]- MU_ghost)/example.dx, label = '20' )
plt.plot(np.zeros(len(CHist[0])), color = 'k')
plt.legend()
plt.show()



if example.jac == True:
    CJAC = []
    AJAC = []
    MUJAC = []
    
    for i in range(example.T_cycle):
        CJAC.append((CHist[i] -Css[0])/example.dx)
        AJAC.append((AHist[i] -AggA[0])/example.dx)
        #MUJAC.append((CHist[i] - MUss[0])/example.dx)


    savemat( 'CJAC_TRAN_hNMPC.mat', mdict = {'CJAC_TRAN_hNMPC': CJAC})
    savemat( 'AJAC_TRAN_hNMPC.mat', mdict = {'AJAC_TRAN_hNMPC': AJAC})
    savemat( 'MUJAC_TRAN_hNMPC.mat', mdict = {'CJAC_TRAN_hNMPC': MUJAC})

    
if example.jacN == True:
    CJACN = []
    AJACN = []
    MUJACN = []
    
    for i in range(example.T_cycle):
        CJACN.append((CHist[i] -Css[0])/example.dx)
        AJACN.append((AHist[i] -AggA[0])/example.dx)
        #MUJACN.append((CHist[i] - MUss[0])/example.dx)

    savemat( 'CJACN_TRAN_hNMPC.mat', mdict = {'CJACN_TRAN_hNMPC': CJACN})
    savemat( 'AJACN_TRAN_hNMPC.mat', mdict = {'AJACN_TRAN_hNMPC': AJACN})
    savemat( 'MUJACN_TRAN_hNMPC.mat', mdict = {'CJACN_TRAN_hNMPC': MUJACN})

    
if example.jacW == True:
    CJACW = []
    AJACW = []
    MUJACW = []
    
    for i in range(example.T_cycle):
        CJACW.append((CHist[i] -Css[0])/example.dx)
        AJACW.append((AHist[i] -AggA[0])/example.dx)
        #MUJACW.append((CHist[i] - MUss[0])/example.dx)
        
    savemat( 'CJACW_TRAN_hNMPC.mat', mdict = {'CJACW_TRAN_hNMPC': CJACW})
    savemat( 'AJACW_TRAN_hNMPC.mat', mdict = {'AJACW_TRAN_hNMPC': AJACW})
    savemat( 'MUJACW_TRAN_hNMPC.mat', mdict = {'CJACW_TRAN_hNMPC': MUJACW})
        
    
    









'''

plt.plot((MUList - MUss[0])/example.dx,label = '15')
plt.plot(np.zeros(len(AggA_List)), color = 'k')
plt.legend()
#plt.savefig("MUJAC_TRANMAT.jpg", dpi=500)
plt.show()



plt.plot((AggA_List - AggA[0])/example.dx,label = '15')
plt.plot(np.zeros(len(AggA_List)), color = 'k')
plt.legend()
#plt.savefig("AJACW.jpg", dpi=500)
plt.show()



plt.plot((AggC_List - Css[0])/example.dx,label = '15',)
plt.plot(np.zeros(len(AggA_List)), color= 'k')
plt.legend()
#plt.savefig("CJACW.jpg", dpi=500)
plt.show()

'''












