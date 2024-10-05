#%%
import numpy as np
import matplotlib.pyplot as plt


class Tissue:
    def __init__(self, num_rows:int, EphA:dict ={}, EphB:dict ={}, efnA:dict ={}, efnB:dict ={}):
        self.name = ''
        x = np.arange(num_rows)
        xx = np.linspace(0,1,num_rows)
         
        self.grid_index = np.array(np.meshgrid(x,x))
        self.grid_pos = np.array(np.meshgrid(xx, xx))

        self.positions = np.array([y for x in self.grid_index.T for y in x]) # list of (x,y) coordinates
        self.axons = np.arange(self.positions.shape[0])
        
        self.EphA = self.sum_grads(EphA, self.grid_pos[0])
        self.EphB = self.sum_grads(EphB, self.grid_pos[1])
        self.efnA = self.sum_grads(efnA, self.grid_pos[0])
        self.efnB = self.sum_grads(efnB, self.grid_pos[1])
        

    def sum_grads(grads:dict, grid_pos): # TODO
        """ take the individual concentration gradients and sum them into a combined gradient"""
        if grads=={}:
            return
        for k, v in grads.items():
            pass

        return LookupError("Function Not Implemented")
    
    

retina = Tissue(6, )
retina.EphA = retina.fractional_pos ** 2


colliculus = Tissue(6,)
colliculus.efnA = colliculus.fractional_pos /2 + 0.5



RCmap = np.random.permutation(colliculus.axons)
RCmap[0]




#%%
length = retina.positions.shape[0]
pairs = np.random.permutation(length).reshape(length//2, 2)

pairs[0]
# %%




