
import numpy as np
from numba import njit

"""_summary_
    - This is going to accept the precalculated positions and reference everything according to the RCmap      
    
Returns:
    _type_: _description_
"""

class Mapper:
    def __init__(self, axons, term_zones, EphA, EphB, efnA, efnB, source_positions, target_positions, alpha, beta, gamma, R, d):
        """ This is going to take the relevant 2D arrays and use them to refine the map"""
        # 1 -- define the lookup tables to be referenced in the comparisson
        self.src_pos = source_positions
        self.trg_pos = target_positions
        self.EphA = EphA
        self.EphB = EphB
        self.efnA = efnA
        self.efnB = efnB
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R = R
        self.d = d
       
        # initialize the random arrangement between the source and target 
        self.RCmap = np.random.permutation(axons.shape[0])
        
        # 2 -- Pair up the Axons 
        pair_list = random_pairs(axons.shape[0])
        
        self.sim_params = {
            'axons': list(axons), 
            'term_zones': list(term_zones),
            'EphA': EphA,
            'efnA': efnA,
            'alpha': alpha,
            'EphB': EphB,
            'efnB': efnB,
            'beta': beta,
            'src_pos': self.src_pos,
            'trg_pos': self.trg_pos, 
            'R': R, 
            'gamma': gamma, 
            'd': d, 
        }

        
    def refine_map(self) -> None:
        RCmap = refine_map_iter(self.RCmap, **self.sim_params)
        return RCmap

@njit
def all_eTotal(pairs, RCmap, axons, term_zones, EphA, efnA, alpha, EphB, efnB, beta, src_pos, trg_pos, R, gamma, d) -> np.ndarray:
    """
    determines if a pair will be swaped stochastically, proportional to eTotal between the pair 

    Returns:
        np.ndarray: defines which pairs will be updated and which will be left alone, according to the energy function
    """
    switch_array =  np.array([pair_eTot(pair,  RCmap, axons, term_zones, EphA, efnA, alpha, EphB, efnB, beta, src_pos, trg_pos, R, gamma, d) for pair in pairs]) 
    return switch_array 

@njit
def pair_eTot(pair, RCmap, axons, term_zones, EphA, efnA, alpha, EphB, efnB, beta, src_pos, trg_pos, R, gamma, d) -> np.ndarray:
    """
    function of the chemical 'energy' and the distance in the source tissue (ret or V1) (representing activity) 

    Returns:
        np.ndarray: the combined chemical and activity defined energies to define if a switch will occour
    """
    ax1, ax2 = pair
    src1, src2  = axons[ax1], axons[ax2]
    tz1, tz2 = term_zones[RCmap[ax1]], term_zones[RCmap[ax2]]
    
    return eChemA(src1, src2, tz1, tz2, EphA, efnA, alpha) + eChemB(src1, src2, tz1, tz2, EphB, efnB, beta) + eAct(src1, src2, tz1, tz2,  src_pos, trg_pos, R, gamma, d) # eTotal

@njit
def swap_pos_sc(pair : np.ndarray, RCmap : np.ndarray, axons: np.ndarray) -> np.ndarray:
    """
    the positions in the sc are swaped given their p-switch (proportional to eTotal)

    Returns:
        np.ndarray: the updated map, with all pairs of axons that meet the threshold, swapped
    """
    ax1, ax2 = pair
    RCmap[ax1], RCmap[ax2] = RCmap[ax2], RCmap[ax1]

    return RCmap

@njit
def update_df(pairs, RCmap, axons) -> np.ndarray:
    """
    swaps the axons if sufficient p-switch
    Returns:
        np.ndarray: the updated map that has been refined
    """
    for pair in pairs:
        RCmap = swap_pos_sc(pair, RCmap, axons)
    return RCmap

@njit
def random_pairs(length) -> np.ndarray:
    """ make a set of random pairs for the eChme+eAct comparisson """
    if length%2:
        length -=1
    return np.random.permutation(length).reshape(length//2, 2)

@njit
def eChemA(ax1, ax2, tz1, tz2, EphA, efnA, alpha) ->  float:
    """
     function of the chemical 'strength' between the two positions as defined by the target efnA and source EphA 

    E = α (RA1 L A2 + RA2 L A1) − α (RA1 L A1 + RA2 L A2) = α (RA1 − RA2)(L A2 − L A1) = α∇ RA∇ L Ax2 (Tsigankov and Koulakov, 2006, p. 106)
    EA = alpha * ( RA(RofL(iL2)) - RA(RofL(iL)) ) * (LA(iL2)-LA(iL)) in Savier 2017 

    Returns:
        np.ndarray: the difference in chemical energy between all pairs of compared axons
    """
    ephA_diff = EphA[ax1[0]][ax1[1]] - EphA[ax2[0]][ax2[1]]
    efnA_diff = efnA[tz2[0]][tz2[1]] - efnA[tz1[0]][tz1[1]]
    return alpha * (ephA_diff) * (efnA_diff) 

@njit
def eChemB(ax1, ax2, tz1, tz2, EphB, efnB, beta) ->  float:
    """
     function of the chemical 'strength' between the two positions as defined by the target efnB and source EphB 

    E = α (RA1 L A2 + RA2 L A1) − α (RA1 L A1 + RA2 L A2) = α (RA1 − RA2)(L A2 − L A1) = α∇ RA∇ L Ax2 (Tsigankov and Koulakov, 2006, p. 106)
    EA = alpha * ( RA(RofL(iL2)) - RA(RofL(iL)) ) * (LA(iL2)-LA(iL)) in Savier 2017 

    Returns:
        np.ndarray: the difference in chemical energy between all pairs of compared axons
    """
    ephB_diff = EphB[ax1[0]][ax1[1]] - EphB[ax2[0]][ax2[1]]
    efnB_diff = efnB[tz1[0]][tz1[1]] - efnB[tz2[0]][tz2[1]]
    return beta * (ephB_diff) * (efnB_diff) # TODO check this and change it to the correct formula for EphB signalling 

@njit
def eAct(ax1, ax2, tz1, tz2,  src_pos, trg_pos, R, gamma, d) ->  np.ndarray:
    """
    function of the distance between the axons in the target (SC) vs in the source tissue (ret or V1)

    Returns:
        np.ndarray: the difference in activity energy between all pairs of compared axons
    """

    source_dist = np.linalg.norm(src_pos[ax1[0]][ax1[1]] - src_pos[ax2[0]][ax1[1]])
    cross_correlation = np.exp(-source_dist/R)              # Cij in Koulakov&Tsigankov 2006
    
    sc_dist = np.linalg.norm(trg_pos[tz1[0]][ax1[1]] - trg_pos[tz2[0]][ax1[1]]) # r in Koulakov&Tsigankov 2006
    form_attract = np.exp(-(sc_dist**2) /(2*d**2))          # U in Koulakov&Tsigankov 2006 - overlap of dendritic arbors (Gaussian) -- 'd' is an experimental value [3%of the SC dimensions, see Methods, Eq. (2)]
    return -(gamma/2) * cross_correlation * form_attract

@njit
def refine_map_iter(RCmap, axons, term_zones, EphA, efnA, alpha, EphB, efnB, beta, src_pos, trg_pos, R, gamma, d, n_repeats=2E2, deterministic=False) -> np.ndarray:
    """
    refines the collicular map N times, iteratively 

    Raises:
        ValueError: The total number of axons must be even, simply due to the mechanism chosen for refinement. All axons are paired with another, and so the total must be even. 

    Returns:
        np.ndarray: the refined map with the residuals of the refinement process (representing how the refinement progressed)
    """
    # Num = len(df[0]) # number of elements to be paired
    # if Num%2: raise ValueError('There must be an even number of neurons to compare')


    for i in range(int(n_repeats)):
        pairs = random_pairs(RCmap.shape[0])                           # makes a complete set of random pairs
        total_energy = all_eTotal(pairs, RCmap, axons, term_zones, EphA, efnA, alpha, EphB, efnB, beta, src_pos, trg_pos, R, gamma, d)                # calcs if eTotal sufficient for swap
        
        norm_eTot = 1 / (1 + np.exp(4 * total_energy))      # normalization of the energy measurements
        if deterministic: 
            comparator =  np.full(len(pairs), 0.5) 
        else: 
            comparator = np.random.random(len(pairs))
        swap = norm_eTot > comparator                   # large Etotal makes a small normalized value i.e., swap would increse map energy

        RCmap = update_df(pairs[swap], RCmap, axons)                     # exchanges the target locations bewteen the pair
    return RCmap


if __name__ == '__main__':
    main()