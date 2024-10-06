#%%
import numpy as np
import matplotlib.pyplot as plt
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

    def sum_grads(self, gradient_dict:dict, normalize=True)-> np.ndarray:
        """ combines the individual gene in a family into a single gradient
                all contributing to the whole

        Returns:
            np.ndarray: the summed gradients
        """
        summed = np.sum(np.array(list(gradient_dict.values())), axis=0)
        summed[summed<0] = 0 # for hypothetical negative 'knockin' mutants

        if normalize:
            # summed = (summed-summed.min())/(summed.max()-summed.min()) + 0.1
            summed = self.normalize_grad(summed)
        return summed
    
    def normalize_grad(self, yy):
        """_summary_

        Args:
            yy (ndarray): The values representing the summed Eph/ephrin gradient. 

        Returns:
            _type_: The array normalized to set the area under the curve to be 1.
        """
        x_spacing = 1/len(yy)
        auc = np.trapz(yy, dx=x_spacing)
        return yy/auc    



retina = Tissue(Num)
x = retina.grid_fract
Num = retina.positions.shape[0]

# There's a problem with the gradients as inserted
ret_EphBs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    'EphA4': 0.04 * np.exp(x[1]) + 0.939, 
    'EphA5': 0.515 * np.exp(x[1]) + 0.1232, 
    'EphA6': 0.572 * np.exp(x[1]) + 0.03
}

ret_EphAs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    'EphA4': 0.04 * np.exp(x[0]) + 0.939 * retina.isl2, 
    'EphA5': 0.515 * np.exp(x[0]) + 0.123, 
    'EphA6': 0.572 * np.exp(x[0]) + 0.03
}

ret_efnAs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    # 'efnA2': (-0.066 * np.exp( -x) +1.045) * 0.11, # only 11% of the efnA5 puncta were counted for efnA2
    'efnA2': (-0.066 * np.exp( -x[0]) +1.045) * 0.11, 
    'efnA3': (0.232 * np.exp(-x[0]) + 0.852)  * 0.22, 
    'efnA5': (1.356 * np.exp(-x[0]) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
}


sc_efnAs_dict = {
    # 'efnA5': -2.365*x**3 + 2.944*x**2 + 0.325*x + 0.454, # polynomial - should arguably use this over the exponential, as even the corrected efnA5 measurement has a fall-off at the posterior-most SC
    'efnA5': 0.646 * np.exp(x[0]) - 0.106, # exponential
    'efnA3': -0.052 * np.exp(x[0]) + 1.008, # exponential
    'efnA2': -0.124*x**3 - 0.896*x**2 + 1.25*x + 0.708, # polynomial
    # 'theoretical': (np.exp((np.arange(Num) - Num) / Num) - np.exp((-np.arange(Num) - Num) / Num))
} # JD Measured

cort_EphAs_dict = {
    'theoretical': (np.exp((np.arange(Num) - Num) / Num) 
                    - np.exp((-np.arange(Num) - Num) / Num))
} # from Savier et al 2017

retina.set_gradients(ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnAs_dict)

colliculus = Tissue(Num)
RCmap = np.random.permutation(colliculus.positions)
RCmap[0]



sim_params = {
    'axons': retina.positions, 
    'term_zones': colliculus.positions,
    'EphA': retina.EphA,
    'efnA': retina.efnA,
    'alpha': 70,
    'EphB': retina.EphB,
    'efnB': retina.efnB.T,
    'beta': 70,
    'source_positions': retina.grid_fract.T,
    'target_positions': retina.grid_fract.T, 
    'R': 0.1, 
    'gamma': 10, 
    'd': 0.3, 
}

m = Mapper(**sim_params)
refined_map = m.refine_map()
refined_map
#%%
plt.imshow(refined_map)
# %%
