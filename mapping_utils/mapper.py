
import numpy as np
from numba import njit

pair = "pairs[0]"

axon1, axon2 = pair

retina = ''
colliculus = ''
RCmap = ''

source = retina
target = colliculus




# %%
topo_map = RCmap

ax1, ax2 = pair
target_pos = target.positions[topo_map[ax1]]
target_efnA = target.efnA[target_pos[0]]

target_pos, target_efnA

source_xy = ''
target_xy = ''


# %%
# TODO need to update these with real numbers
alpha = 1
beta = 1
gamma = 1 
d = 1
R = 1





@njit
def eChem_A(ax1, ax2, df, alpha=alpha) ->  np.ndarray:
    """
    EphA/ephrin-A
    """
    eph_diff = source.EphA[ax1] - source.EphA[ax2]
    efn_diff = target.efnA[target.positions[topo_map[ax2]][0]] - target.efnA[target.positions[topo_map[ax1]][0]] # this converts the topographic map into an efnA concentation at the target for axon1 and axon2
    return alpha * (eph_diff) * (efn_diff) 

@njit
def eChem_B(ax1, ax2, df, beta=beta) ->  np.ndarray:
    """
    EphB/ephrin-B
    """
    eph_diff = source.EphB[ax1] - source.EphB[ax2]
    efn_diff = target.efnB[target.positions[topo_map[ax2]][0]] - target.efnB[target.positions[topo_map[ax1]][0]] # this converts the topographic map into an efnB concentation at the target for axon1 and axon2
    return beta * (eph_diff) * (efn_diff) 
@njit
def eAct(ax1 :  np.ndarray, ax2 : np.ndarray,  df : np.ndarray) ->  np.ndarray:
    """
    function of the distance between the axons in the target (SC) vs in the source tissue (ret or V1)

    Returns:
        np.ndarray: the difference in activity energy between all pairs of compared axons
    """
    source_dist = np.abs(df[0,ax1] - df[0,ax2])
    cross_correlation = np.exp(-source_dist/R)              # Cij in Koulakov&Tsigankov 2006
    
    sc_dist = np.abs(df[1,ax1] - df[1,ax2])                 # r in Koulakov&Tsigankov 2006
    form_attract = np.exp(-(sc_dist**2) /(2*d**2))          # U in Koulakov&Tsigankov 2006 - overlap of dendritic arbors (Gaussian) -- 'd' is an experimental value [3%of the SC dimensions, see Methods, Eq. (2)]
    return -(gamma/2) * cross_correlation * form_attract

@njit
def pair_eTot(pair : np.ndarray, df : np.ndarray) -> np.ndarray:
    """
    function of the chemical 'energy' and the distance in the source tissue (ret or V1) (representing activity) 

    Args:
        pair (np.ndarray): the two axons that are to be checked
        df (np.ndarray): the entire map that is to be refined

    Returns:
        np.ndarray: the combined chemical and activity defined energies to define if a switch will occour
    """
    ax1, ax2 = pair
    return eChem_A(ax1, ax2, df) + eChem_B(ax1, ax2, df) + eAct(ax1, ax2, df) # eTotal

@njit
def swap_pos_sc(pair : np.ndarray, df : np.ndarray) -> np.ndarray:
    """
    the positions in the sc are swaped given their p-switch (proportional to eTotal)

    Args:
        pair (np.ndarray): the pair of axons to be swapped
        df (np.ndarray): the entire map that is to be refined

    Returns:
        np.ndarray: the updated map, with all pairs of axons that meet the threshold, swapped
    """
    ax1, ax2 = pair
    df[1, ax1], df[1, ax2] = df[1, ax2], df[1, ax1] # swap
    return df