
import numpy as np
import matplotlib.pyplot as plt

"""_summary_
    - This is going to accept the precalculated positions and reference everything according to the RCmap      
    
Returns:
    _type_: _description_
"""

class Mapper:
    def __init__(self, axons:np.array, refinement_map:np.array, EphA:np.array, EphB:np.array, efnA:np.array, efnB:np.array, source_positions:np.array, target_positions:np.array, alpha, beta, gamma, R, d):
        """ This is going to take the relevant 2D arrays and use them to refine the map"""
        # 1 -- define the lookup tables to be referenced in the comparisson
        self.axons = axons
        self.ref_map = refinement_map
        self.EphA = EphA
        self.efnA = efnA
        self.EphB = EphB
        self.efnB = efnB
        self.src_pos = source_positions
        self.trg_pos = target_positions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R = R
        self.d = d
       
        # initialize the random arrangement between the source and target 
        self.RCmap = np.random.permutation(axons.shape[0])
        
        # 2 -- Pair up the Axons 
        pair_list = random_pairs(axons.shape[0])
        
        sim_params = {
            'RCmap': self.RCmap,
            'EphA': self.EphA,
            'efnA': self.efnA,
            'alpha': self.alpha,
            'EphB': self.EphB,
            'efnB': self.efnB,
            'beta': self.beta,
            'src_pos': self.src_pos,
            'trg_pos': self.trg_pos, 
            'R': self.R, 
            'gamma': self.gamma, 
            'd': self.d, 
        }
        # 3 -- check them according to the refinement algorithm
        
        # 4 -- decide which pairs to swap and which to keep
        
        # 5 -- update the RCmap
        
        pass

@njit
def all_eTotal(pairs, sim_params) -> np.ndarray:
    """
    determines if a pair will be swaped stochastically, proportional to eTotal between the pair 

    Args:
        pairs (_type_): the pairs of axons to be tested 
        df (_type_): the entire map to be refined 

    Returns:
        np.ndarray: defines which pairs will be updated and which will be left alone, according to the energy function
    """
    switch_array =  np.array([pair_eTot(pair, **sim_params) for pair in pairs]) 
    return switch_array 

@njit
def pair_eTot(pair : np.ndarray, RCmap, EphA, efnA, alpha, EphB, efnB, beta, src_pos, trg_pos, R, gamma, d, **sim_params) -> np.ndarray:
    """
    function of the chemical 'energy' and the distance in the source tissue (ret or V1) (representing activity) 

    Returns:
        np.ndarray: the combined chemical and activity defined energies to define if a switch will occour
    """
    ax1, ax2 = pair
    return eChemA(RCmap, ax1, ax2, EphA, efnA, alpha) + eChemB(RCmap, ax1, ax2, EphB, efnB, beta) + eAct(RCmap, ax1, ax2,  src_pos, trg_pos, R, gamma, d) # eTotal




@njit
def random_pairs(length) -> np.ndarray:
    """ make a set of random pairs for the eChme+eAct comparisson """
    if length%2:
        length -=1
    return np.random.permutation(length).reshape(length//2, 2)

@njit
def eChemA(RCmap: np.ndarray, ax1 : np.ndarray, ax2 : np.ndarray, EphA : np.ndarray, efnA: np.ndarray, alpha:int) ->  float:
    """
     function of the chemical 'strength' between the two positions as defined by the target efnA and source EphA 

    E = α (RA1 L A2 + RA2 L A1) − α (RA1 L A1 + RA2 L A2) = α (RA1 − RA2)(L A2 − L A1) = α∇ RA∇ L Ax2 (Tsigankov and Koulakov, 2006, p. 106)
    EA = alpha * ( RA(RofL(iL2)) - RA(RofL(iL)) ) * (LA(iL2)-LA(iL)) in Savier 2017 

    Returns:
        np.ndarray: the difference in chemical energy between all pairs of compared axons
    """
    inv_map = RCmap.argsort()
    
    ephA_diff = EphA[ax1] - EphA[ax2]
    efnA_diff = efnA[RCmap[ax1]] - efnA[RCmap[ax1]]
    return alpha * (ephA_diff) * (efnA_diff) 

@njit
def eChemB(RCmap: np.ndarray, ax1 : np.ndarray, ax2 : np.ndarray, EphB : np.ndarray, efnB: np.ndarray, beta:int) ->  float:
    """
     function of the chemical 'strength' between the two positions as defined by the target efnB and source EphB 

    E = α (RA1 L A2 + RA2 L A1) − α (RA1 L A1 + RA2 L A2) = α (RA1 − RA2)(L A2 − L A1) = α∇ RA∇ L Ax2 (Tsigankov and Koulakov, 2006, p. 106)
    EA = alpha * ( RA(RofL(iL2)) - RA(RofL(iL)) ) * (LA(iL2)-LA(iL)) in Savier 2017 

    Returns:
        np.ndarray: the difference in chemical energy between all pairs of compared axons
    """
    ephB_diff = EphB[ax1] - EphB[ax2]
    efnB_diff = efnB[RCmap[ax1]] - efnB[RCmap[ax1]]
    print("EphB formula needs to be inserted")
    return beta * (ephB_diff) * (efnB_diff) # TODO check this and change it to the correct formula for EphB signalling 

@njit
def eAct(RCmap, ax1, ax2,  src_pos, trg_pos, R:float, gamma:float, d:float) ->  np.ndarray:
    """
    function of the distance between the axons in the target (SC) vs in the source tissue (ret or V1)

    Returns:
        np.ndarray: the difference in activity energy between all pairs of compared axons
    """
    
    source_dist = np.abs(src_pos[ax1] - src_pos[ax2])
    cross_correlation = np.exp(-source_dist/R)              # Cij in Koulakov&Tsigankov 2006
    
    sc_dist = np.abs(trg_pos[RCmap[ax1]] - trg_pos[RCmap[ax2]]) # r in Koulakov&Tsigankov 2006
    form_attract = np.exp(-(sc_dist**2) /(2*d**2))          # U in Koulakov&Tsigankov 2006 - overlap of dendritic arbors (Gaussian) -- 'd' is an experimental value [3%of the SC dimensions, see Methods, Eq. (2)]
    return -(gamma/2) * cross_correlation * form_attract