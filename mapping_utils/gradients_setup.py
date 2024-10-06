#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mapper import Mapper, refine_map_iter
Num = 100

class Tissue:
    def __init__(self, num_rows:int):
        self.name = ''
        x = np.arange(num_rows)
        xx = np.linspace(0,1,num_rows)
        
        # two ways of representing the same data. Not sure which is the propper one 
        self.grid_index = np.array(np.meshgrid(x,x)) # a 2D grid of indices for a grid of NxN neurons
        self.grid_fract = np.array(np.meshgrid(xx, xx)) # a 2D grid of spatial coordinates from 0-1

        self.positions = np.array([y for x in self.grid_index.T for y in x]) # array of (x,y) indices
        
        
        self.isl2 = np.random.randint(0, 2, (num_rows, num_rows)) # 2D array of Isl2+ cells (technically only available in the retina)

        
    def set_gradients(self, EphA:dict ={}, EphB:dict ={}, efnA:dict ={}, efnB:dict ={}):
        self.EphA = self.sum_grads(EphA)
        self.EphB = self.sum_grads(EphB)
        self.efnA = self.sum_grads(efnA)
        self.efnB = self.sum_grads(efnB)

    def sum_grads(self, gradient_dict:dict)-> np.ndarray:
        """ combines the individual gene in a family into a single gradient
                all contributing to the whole

        Returns:
            np.ndarray: the summed gradients
        """
        return sum_grads_list(np.array(list(gradient_dict.values())))
        
        
        
        
# @njit       
def sum_grads_list(grad_list):
    summed = grad_list[0]
    for grad in grad_list[1:]:
        summed+= grad
    summed[summed<0] = 0 # for hypothetical negative 'knockin' mutants
    # summed = normalize_grad(summed) 
    return summed

@njit
def normalize_grad(yy): # TODO this normalizing strategy doesn't work for the 2D gradients -- this needs to be coded differently -- normalizing the gradients before stretching them out could be a slution, but that would need come way to incorporate specific Isl2 insertions/deletions for specific members, after they have been summed and turned 2D
    """_summary_

    Args:
        yy (ndarray): The values representing the summed Eph/ephrin gradient. 

    Returns:
        _type_: The array normalized to set the area under the curve to be 1.
    """
    x_spacing = 1/yy.shape[0]
    auc = np.trapz(yy, dx=x_spacing)
    return yy/auc    



retina = Tissue(Num)

x = retina.grid_fract

# There's a problem with the gradients as inserted
ret_EphBs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    'EphA4': 0.04 * np.exp(x[1]) + 0.939, 
    'EphA5': 0.515 * np.exp(x[1]) + 0.1232, 
    'EphA6': 0.572 * np.exp(x[1]) + 0.03
}

ret_EphAs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    'EphA4': 0.04 * np.exp(x[0]) + 0.939, #* retina.isl2, 
    'EphA5': 0.515 * np.exp(x[0]) + 0.123, 
    'EphA6': 0.572 * np.exp(x[0]) + 0.03
}

ret_efnBs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    # 'efnA2': (-0.066 * np.exp( -x) +1.045) * 0.11, # only 11% of the efnA5 puncta were counted for efnA2
    'efnA2': (-0.066 * np.exp( -x[1]) +1.045) * 0.11, 
    'efnA3': (0.232 * np.exp(-x[1]) + 0.852)  * 0.22, 
    'efnA5': (1.356 * np.exp(-x[1]) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
}

ret_efnAs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    # 'efnA2': (-0.066 * np.exp( -x) +1.045) * 0.11, # only 11% of the efnA5 puncta were counted for efnA2
    'efnA2': (0.066 * np.exp( -x[0]) +1.045) * 0.11, 
    'efnA3': (0.232 * np.exp(-x[0]) + 0.852)  * 0.22, 
    'efnA5': (1.356 * np.exp(-x[0]) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
}

sc_efnAs_dict = {
    # 'efnA5': -2.365*x**3 + 2.944*x**2 + 0.325*x + 0.454, # polynomial - should arguably use this over the exponential, as even the corrected efnA5 measurement has a fall-off at the posterior-most SC
    'efnA5': 0.646 * np.exp(x[1]) - 0.106, # exponential
    'efnA3': -0.052 * np.exp(x[1]) + 1.008, # exponential
    'efnA2': -0.124*x[1]**3 - 0.896*x[1]**2 + 1.25*x[1] + 0.708, # polynomial
    # 'theoretical': (np.exp((np.arange(Num) - Num) / Num) - np.exp((-np.arange(Num) - Num) / Num))
} # JD Measured

cort_EphAs_dict = {
    'theoretical': (np.exp((np.arange(Num) - Num) / Num) 
                    - np.exp((-np.arange(Num) - Num) / Num))
} # from Savier et al 2017
#%%

retina.set_gradients(ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict)

colliculus = Tissue(Num)
colliculus.set_gradients(ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict)
# RCmap = np.random.permutation(colliculus.positions.size[0])
# RCmap[0]

#%%
plt.imshow(retina.EphB)
plt.show()
plt.imshow(retina.EphA)

#%%

def make_map_df(RCmap, retina, colliculus):
    
    id_src = np.arange(retina.positions.shape[0])
    EphA_at_src = np.array([retina.EphA[*x] for x in retina.positions])
    EphB_at_src = np.array([retina.EphB[*x] for x in retina.positions])
    pos_at_src = np.array([retina.grid_fract.T[*x] for x in retina.positions])
    
    RCmap = RCmap
    efnA_at_trg = np.array([colliculus.efnA[*x] for x in colliculus.positions[RCmap]])
    efnB_at_trg = np.array([colliculus.efnB[*x] for x in colliculus.positions[RCmap]])
    pos_at_trg = np.array([colliculus.grid_fract.T[*x] for x in colliculus.positions[RCmap]])
    
    return np.vstack((id_src, EphA_at_src, EphB_at_src, pos_at_src.T[0], pos_at_src.T[1], RCmap, efnA_at_trg, efnB_at_trg, pos_at_trg.T[0], pos_at_trg.T[1]))

df = make_map_df(np.random.permutation(Num**2), retina, colliculus)
df        # SourceID, EphA, EphB, RetX, RetY, RCmap, efnA, efnB, scX, scY

#%%       
        
    
















sim_params = {
    'axons': retina.positions, 
    'term_zones': colliculus.positions,
    'EphA': retina.EphA,
    'efnA': retina.efnA,
    'alpha': 70,
    'EphB': retina.EphB,
    'efnB': retina.efnB,
    'beta': 70,
    'source_positions': retina.grid_fract,
    'target_positions': retina.grid_fract, 
    'R': 0.1, 
    'gamma': 10, 
    'd': 0.3, 
}

# TODO Build a dataframe that can be passed to the mapper script 
#   - it needs to be able to accept the calculated values for each axon's EpHA, efnA, B, b, position in the source, and position in the target.


#%%



m = Mapper(**sim_params)
refined_map = m.refine_map()
refined_map
#%%
plt.imshow(refined_map)
# %%
