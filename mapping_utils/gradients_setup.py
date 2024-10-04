#%%
import numpy as np
import matplotlib.pyplot as plt


class Tissue:
    def __init__(self, num_rows, grad_EphA=1, grad_efnA=1, grad_EphB=1, grad_efnB=1):
        self.name = ''
        x = np.arange(num_rows)
        self.grid = np.meshgrid(x,x)
        self.grid = np.array(self.grid)

        self.positions = np.array([y for x in self.grid.T for y in x]) # list of (x,y) coordinates
        self.axons = np.arange(self.positions.shape[0])
        
                


retina = Tissue(6,)
colliculus = Tissue(6,)


retina.positions
RCmap = np.random.permutation(colliculus.axons)

RCmap[0]



pairs = np.array(np.split(retina.axons, 2)).T


pairs[0]
# %%
