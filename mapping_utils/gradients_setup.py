#%%
import numpy as np
import matplotlib.pyplot as plt


class Tissue:
    def __init__(self, num_rows, grad_EphA=1, grad_efnA=1, grad_EphB=1, grad_efnB=1):
        self.name = ''
        x = np.arange(num_rows)
        self.fractional_pos = np.linspace(0,1,num_rows)
         
        self.grid = np.meshgrid(x,x)
        self.grid = np.array(self.grid)

        self.positions = np.array([y for x in self.grid.T for y in x]) # list of (x,y) coordinates
        self.axons = np.arange(self.positions.shape[0])
        

retina = Tissue(6,)
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




