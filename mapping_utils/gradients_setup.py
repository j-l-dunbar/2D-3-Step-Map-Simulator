#%%
import numpy as np
import matplotlib.pyplot as plt



class Tissue:
    def __init__(self, num_rows:int, EphA:dict ={}, EphB:dict ={}, efnA:dict ={}, efnB:dict ={}):
        self.name = ''
        x = np.arange(num_rows)
        xx = np.linspace(0,1,num_rows)
        
        # two ways of representing the same data. Not sure which is the propper one 
        self.grid_index = np.array(np.meshgrid(x,x)) # a 2D grid of indices for a grid of NxN neurons
        self.grid_fract = np.array(np.meshgrid(xx, xx)) # a 2D grid of spatial coordinates from 0-1

        self.positions = np.array([y for x in self.grid_index.T for y in x]) # array of (x,y) indices
        
        self.EphA = self.sum_grads(EphA, self.grid_fract[0])
        self.EphB = self.sum_grads(EphB, self.grid_fract[1])
        self.efnA = self.sum_grads(efnA, self.grid_fract[0])
        self.efnB = self.sum_grads(efnB, self.grid_fract[1])
        
        self.isl2 = np.random.randint(0, 2, (num_rows, num_rows)) # 2D array of Isl2+ cells (technically only available in the retina)
        

    def sum_grads(self, grads:dict, grid_pos): # TODO
        """ take the individual concentration gradients and sum them into a combined gradient"""
        if grads=={}:
            return None
        for k, v in grads.items():
            pass # TODO look at old 1D code to do this. it should be exactly the same

        return LookupError("Function Not Implemented")
    
    
ret_EphA = {}
ret_EphB = {}
ret_efnA = {}
ret_efnB = {}

retina = Tissue(100, ret_EphA, ret_EphB, ret_efnA, ret_efnB)

test_gradient = retina.grid_fract[0]**1.2 * retina.isl2

plt.imshow(test_gradient) # + retina.grid_fract[1]**1.5) # Messing around showing what superimposed gradients of A and Bs with Isl2 would look like
# TODO need to figure out a good scheme for processing the mutants 
#       - need to add a function to delete specific targets, both with Isl2, and constitutively
#       - need to make a presentable array of 
#           - the positions in the retina
#           - the EphA&B in the retina
#           - positions in the SC
#           - the efnA & B in the SC
#      This all needs to be processed in such a way that it is easy to reference for the refinement algo




#%%


col_grads = {}
colliculus = Tissue(6,col_grads)



RCmap = np.random.permutation(colliculus.axons)
RCmap[0]




#%%
length = retina.positions.shape[0]
pairs = np.random.permutation(length).reshape(length//2, 2)

pairs[0]


for pair in pairs:
    ax1, ax2 = pair
    
    ret1, ret2 = 'positions in the retina', '' # TODO need some fucntion that will take an axon's index and retun a numerical position in the retina
    col1, col2 = 'positions in the colliculus', '' # TODO need some fucntion that will take an axon's index and retun a numerical position in the colliculus
    
    EphA1, EphA2 = 'function of position in retina', '' # TODO need some function that will get the relevant EphAs
    efnA1, efnA2 = 'function of position in SC', '' # TODO same as above

    EphB1, EphB2 = 'function of position in retina', ''  # TODO same as above
    efnB1, efnB2 = 'function of position in SC', '' # TODO same as above

# %%



def get_frac_pos(axon_list, target_df):
    pass
