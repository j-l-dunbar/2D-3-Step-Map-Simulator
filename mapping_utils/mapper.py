"""defines the parameters for a mutant condition and simulates the refined topographic connections between a source and target tissue that will result, based on mathematical modeling done by Tsigankov and Koulakov. 

Returns:
    np.ndarray: the refined map phenotype, defined as follows:
                 'id_src': 0, # index column 
            'EphA_at_src': 1, # [EphA] this axons carries from the source tissue     
            'EphB_at_src': 2, # [EphB] this axons carries from the source tissue
        'pos_at_src.T[0]': 3, # source X coords (decimal)
        'pos_at_src.T[1]': 4, # source Y coods (decimal)
                  'RCmap': 5, # how this axons connects to the SC
            'efnA_at_trg': 6, # [efnA] that this axons sees in the target
            'efnB_at_trg': 7, # [efnB] that this axons sees in the target
        'pos_at_trg.T[0]': 8, # target X coords
        'pos_at_trg.T[1]': 9, # target Y coords
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from fractions import Fraction


class Tissue:
    def __init__(self, num_rows:int, name=None):
        """ defines the biological properties of various tissues to be used in the simulation of the retino-collicular and cortico-collicular topographic maps

        Args:
            num_rows (int): sqrt of the number of neurons/axons to be simulated in the topgographic map
            name (string, optional): name for the tissue defined. Defaults to None.
        """
        self.name = name
        self.Num = num_rows
        
        x = np.arange(num_rows)
        xx = np.linspace(0,1,num_rows)
        
        # two ways of representing the same data. Not sure which is the propper one 
        self.grid_index = np.array(np.meshgrid(x,x)) # a 2D grid of indices for a grid of NxN neurons
        self.grid_fract = np.array(np.meshgrid(xx, xx)) # a 2D grid of spatial coordinates from 0-1

        self.positions = np.array([y for x in self.grid_index.T for y in x]) # array of (x,y) indices
        
        self.isl2 = np.random.randint(0, 2, (num_rows, num_rows)) # 2D array of Isl2+ cells (technically only available in the retina)
        self.isl2_hetko = (1+self.isl2)/2 # zeros become 0.5 - for single allele mutants

        self.EphA_dict, self.EphB_dict = None, None
        self.efnA_dict, self.efnB_dict = None, None

    def set_gradients(self):
        ''' defines the expression gradients to be use for simulation a map phenotype ''' 
        if self.EphA_dict: self.EphA = self.sum_grads(self.EphA_dict)
        if self.EphB_dict: self.EphB = self.sum_grads(self.EphB_dict)
        if self.efnA_dict: self.efnA = self.sum_grads(self.efnA_dict)
        if self.efnB_dict: self.efnB = self.sum_grads(self.efnB_dict)

    def sum_grads(self, gradient_dict:dict)-> np.ndarray:
        """ combines the individual gene in a family into a single gradient
                all contributing to the whole

        Returns:
            np.ndarray: the summed gradients
        """
        return self.sum_grads_list(np.array(list(gradient_dict.values())))
        
    def sum_grads_list(self, grad_list):
        """takes a series of expression profiles and combines them into a single 2D expression profile
        """
        summed = grad_list[0]
        for grad in grad_list[1:]:
            summed+= grad
        summed[summed<0] = 0 # to accound for hypothetical negative 'knockin' mutants (i.e., knockdown mutants)
        # summed = self.normalize_grad(summed) #TODO nomalizing the 2D gradients should be implemented
        return summed

    def normalize_grad(self, xy): # TODO this normalizing strategy doesn't work for the 2D gradients -- this needs to be coded differently -- normalizing the gradients before stretching them out could be a slution, but that would need come way to incorporate specific Isl2 insertions/deletions for specific members, after they have been summed and turned 2D
        """attempting to normalize the 2D expression gradients. Not yet implemented
        """
        # for i, yy in enumerate(xy): # this only works for 1D arrays... 
        #     x_spacing = 1/yy.shape[0]
        #     auc = np.trapz(yy, dx=x_spacing)
        #     xy[i] = yy/auc  
        # return xy  

        # return xy/np.max(xy) 
        raise NotImplementedError("Normalization is not properly implemented now.") # TODO find a better way to standardize the "Area Under the Curve" of the summed gradientes into a standard total value

    def make_std_grads(self, EphA_angle=90, EphB_angle=180, efnA_angle=270, efnB_angle=0):
        """ defines the gradients used to simulate a given mutant topographic map
        here they are defined as a simple 1D function stretched across the tissue axis into 2D, but any 'image' representing a 2D expression profile will work just as well to feed into the simulation

        Args:
            EphA_angle (int, optional): the angle at which the EphA gradients run relative to the tissue axes. Defaults to 90.
            EphB_angle (int, optional): the angle at which the EphB gradients run relative to the tissue axes. Defaults to 180.
            efnA_angle (int, optional): the angle at which the efnA gradients run relative to the tissue axes. Defaults to 270.
            efnB_angle (int, optional): the angle at which the efnB gradients run relative to the tissue axes. Defaults to 0.

        Returns:
            [np.ndarray, ...]: a series of arrays representing the graded expression of axon guidance molecules
        """
        x= self.grid_fract

        EphA_comps = np.cos(EphA_angle*np.pi/180)*x[1] + np.sin(EphA_angle*np.pi/180)*x[0]
        EphB_comps = np.cos(EphB_angle*np.pi/180)*x[1] + np.sin(EphB_angle*np.pi/180)*x[0]
        efnA_comps = np.cos(efnA_angle*np.pi/180)*x[1] + np.sin(efnA_angle*np.pi/180)*x[0]
        efnB_comps = np.cos(efnB_angle*np.pi/180)*x[1] + np.sin(efnB_angle*np.pi/180)*x[0]

        # There's a problem with the gradients as inserted
        ret_EphAs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
            'EphA4': 0.040 * np.exp(EphA_comps) + 0.939 , 
            'EphA5': 0.515 * np.exp(EphA_comps) + 0.1232, 
            'EphA6': 0.572 * np.exp(EphA_comps) + 0.03
        }
        ret_efnAs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
            'efnA2': (0.066 * np.exp(efnA_comps) + 1.045) * 0.11, 
            'efnA3': (0.232 * np.exp(efnA_comps) + 0.852) * 0.22, 
            'efnA5': (1.356 * np.exp(efnA_comps) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
        }
        
        ret_EphBs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
            'EphBa': 0.040 * np.exp(EphB_comps) + 0.939 , 
            'EphBb': 0.515 * np.exp(EphB_comps) + 0.123, 
            'EphBc': 0.572 * np.exp(EphB_comps) + 0.03
        }
        ret_efnBs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
            'efnBa': (-0.066 * np.exp(efnB_comps) +1.045) * 0.11, 
            'efnBb': ( 0.232 * np.exp(efnB_comps) + 0.852)  * 0.22, 
            'efnBc': ( 1.356 * np.exp(efnB_comps) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
        }
        sc_efnAs_dict = {
            # 'efnA5': -2.365*x**3 + 2.944*x**2 + 0.325*x + 0.454, # polynomial - should arguably use this over the exponential, as even the corrected efnA5 measurement has a fall-off at the posterior-most SC
            'efnA5': 0.646 * np.exp(efnA_comps) - 0.106, # exponential
            'efnA3': -0.052 * np.exp(efnA_comps) + 1.008, # exponential
            'efnA2': -0.124*(efnA_comps)**3 - 0.896*(efnA_comps)**2 + 1.25*(efnA_comps) + 0.708, # polynomial
        } # JD Measured
        
        sc_efnBs_dict = {
            'theoretical': 1 - np.tile(np.exp((np.arange(self.Num) - self.Num) / self.Num) 
                            - np.exp((-np.arange(self.Num) - self.Num) / self.Num), [self.Num, 1])
        }
        cort_EphAs_dict = {
            'theoretical': np.tile(np.exp((np.arange(self.Num) - self.Num) / self.Num) 
                            - np.exp((-np.arange(self.Num) - self.Num) / self.Num), [self.Num, 1])
        } # from Savier et al 2017
        cort_EphBs_dict = {
            'theoretical': np.tile(np.exp((np.arange(self.Num) - self.Num) / self.Num) 
                            - np.exp((-np.arange(self.Num) - self.Num) / self.Num), [self.Num, 1]).T
        } # from Savier et al 2017
        
        return ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict, sc_efnAs_dict, sc_efnBs_dict, cort_EphAs_dict, cort_EphBs_dict
    
    def make_isl2_ki(self, mutant_name:str, strength:float, target_dict:dict, het:bool=False):
        ''' defines a conditional knockin under control of isl2 (i.e., in 50% of the projecting axons)'''
        if mutant_name in target_dict.keys(): 
            raise NameError("That gene name has already been asigned.")        
        isl2 = self.isl2
        if het: isl2 *= 0.5 
        figure_title = f"{mutant_name} - {'ki/+' if het else 'ki/ki'} - {strength}" 
        target_dict[mutant_name] = strength * isl2
        return figure_title, target_dict

    def make_isl2_ko(self, mutant_name:str, target_dict:dict, het:bool=False):
        ''' defines a conditional knockout under control of isl2 (i.e., in 50% of the projecting axons)'''
        if mutant_name not in target_dict.keys(): 
            raise NameError("That gene has not been defined")
        isl2 = self.isl2
        if het: isl2 = self.isl2_hetko
        figure_title = f"{mutant_name} {'ko/+' if het else 'ko/ko'}" 
        target_dict[mutant_name] *= isl2
        return figure_title, target_dict




class Mapper:
    def __init__(self, alpha=60, beta=60, gamma=120, R=0.11, d=0.03, Num=100, **_):
        """ sets up the params for map refinement before simulating a given mutant condition. Map is refined based on mathematical modeling done by Koulakov and Tsigankov. """
        # 1 -- define the mapping params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R = R 
        self.d = d        
        self.Num = Num
        x = np.arange(Num)
        self.grid_index = np.array(np.meshgrid(x,x)) # a 2D grid of indices for a grid of NxN neurons
        self.positions = np.array([y for x in self.grid_index.T for y in x])
        
    def init_random_map(self):
        """ a randomized 1D array representing the initial unrefined map between loci in the source tissue and loci in the target tissue
        """
        return np.random.permutation(self.Num**2)
    
    def make_map_df(self, hash_map, src_tissue, trg_tissue):    
        ''' takes the various arrays and combines them into a single dataframe for Numba to opperate on efficiently 
                - the abstraction of doing it this way is worth if for the speed that is gained
        '''
        df_index = { # definiton of the columns that are returned in the np.ndarray at the end of the simulation
                 'id_src': 0, # index column 
            'EphA_at_src': 1, # [EphA] this axons carries
            'EphB_at_src': 2, # [EphB] this axons carries
        'pos_at_src.T[0]': 3, # source X (decimal)
        'pos_at_src.T[1]': 4, # source Y (decimal)
                  'RCmap': 5, # how this axons connects to the SC
            'efnA_at_trg': 6, # [efnA] that this axons sees
            'efnB_at_trg': 7, # [efnB] that this axons sees
        'pos_at_trg.T[0]': 8, # target X
        'pos_at_trg.T[1]': 9, # target Y
        }
        
        id_src = np.arange(src_tissue.positions.shape[0])
        EphA_at_src = np.array([src_tissue.EphA[*x] for x in src_tissue.positions])
        EphB_at_src = np.array([src_tissue.EphB[*x] for x in src_tissue.positions])
        pos_at_src = np.array([src_tissue.grid_fract.T[*x] for x in src_tissue.positions])
        
        efnA_at_trg = np.array([trg_tissue.efnA[*x] for x in trg_tissue.positions[hash_map]])
        efnB_at_trg = np.array([trg_tissue.efnB[*x] for x in trg_tissue.positions[hash_map]])
        pos_at_trg = np.array([trg_tissue.grid_fract.T[*x] for x in trg_tissue.positions[hash_map]])
        
        self.df = np.vstack((id_src, EphA_at_src, EphB_at_src, pos_at_src.T[0], pos_at_src.T[1], hash_map, efnA_at_trg, efnB_at_trg, pos_at_trg.T[0], pos_at_trg.T[1]))
        return self.df

    def df_show_grads(self):
        """visualizes the gradients used for simulating a topographic map phenotype
        """
        df = self.df
        EphA, EphB = df[1:3]
        efnA, efnB = df[6:8][:,df[5].argsort()]
        arrs = [EphA, efnA, EphB, efnB]
        colors = ['GnBu', 'GnBu', 'OrRd', 'OrRd']
        
        fig, axes = plt.subplots(figsize=(9,9), ncols = 2, nrows=2)
        axes = axes.flat
        
        for ax, arr, col in zip(axes, arrs, colors):
            empty_img = np.zeros((self.Num, self.Num))
            for i, xy in zip(arr, self.positions):
                empty_img[*xy] = i
            ax.imshow(empty_img, cmap=col, origin='lower')
            self.fractional_axes([ax], self.Num)
            
        return fig

    def fractional_axes(self, axes, Num, color='k', numticks=9):
        for ax in axes:
            ax.set_xlim(0, Num)
            ax.set_ylim(0, Num)
            ax.xaxis.set_major_locator(tk.LinearLocator(numticks))
            ax.yaxis.set_major_locator(tk.LinearLocator(numticks))

            x = ax.get_xticks()
            y = ax.get_yticks()

            x_labs = ["{}/{}".format(*Fraction(i/Num).limit_denominator(16).as_integer_ratio()) if i>0 else '0' for i in x]
            y_labs = ["{}/{}".format(*Fraction(i/Num).limit_denominator(16).as_integer_ratio()) if i>0 else '0' for i in y]
            x_labs[-1] = 1
            y_labs[-1] = 1
            ax.set_xticklabels(x_labs)
            ax.set_yticklabels(y_labs)
            ax.grid(c=color)
        return axes

    def refine_map(self, n_repeats=2E4, deterministic=True) -> None:
        print('Refining map...')
        df = refine_map_iter(self.df, self.alpha, self.beta, 
                             self.gamma, self.R, self.d, 
                             n_repeats=n_repeats, 
                             deterministic=deterministic, 
                             parallel=True)
        return df

@njit(nogil=True)
def all_eTotal(pairs, df, alpha, beta, gamma, R, d) -> np.ndarray:
    """
    determines if a pair will be swaped stochastically, proportional to eTotal between the pair 

    Returns:
        np.ndarray: defines which pairs will be updated and which will be left alone, according to the energy function
    """
    switch_array =  np.array([pair_eTot(pair, df, alpha, beta, gamma, R, d) for pair in pairs]) 
    return switch_array 

@njit(nogil=True)
def pair_eTot(pair, df, alpha, beta, gamma, R, d) -> np.ndarray:
    """
    function of the chemical 'energy' and the distance in the source tissue (ret or V1) (representing activity) 

    Returns:
        np.ndarray: the combined chemical and activity defined energies to define if a switch will occour
    """   
    ax1, ax2 = pair
    return eChemA(ax1, ax2, df, alpha) - eChemB(ax1, ax2, df, beta) + eAct(ax1, ax2, df, R, gamma, d) # eTotal


@njit(nogil=True)
def swap_pos_sc(pair : np.ndarray, df : np.ndarray) -> np.ndarray:
    """
    the positions in the sc are swaped given their p-switch (proportional to eTotal)

    Returns:
        np.ndarray: the updated map, with all pairs of axons that meet the threshold, swapped
    """
    ax1, ax2 = pair
    for i in range(5,10):
        df[i,ax1], df[i,ax2] = df[i,ax2], df[i,ax1]

    return df

@njit(nogil=True)
def update_df(pairs, df) -> np.ndarray:
    """
    swaps the axons if sufficient p-switch
    Returns:
        np.ndarray: the updated map that has been refined
    """
    for pair in pairs:
        df = swap_pos_sc(pair, df)
    return df

@njit(nogil=True)
def random_pairs(length) -> np.ndarray:
    """ make a set of random pairs for the eChme+eAct comparisson """
    if length%2:
        length -=1
    return np.random.permutation(length).reshape(length//2, 2)

@njit(nogil=True)
def eChemA(ax1, ax2, df, alpha) ->  float:
    """
     function of the chemical 'strength' between the two positions as defined by the target efnA and source EphA 

    E = α (RA1 L A2 + RA2 L A1) − α (RA1 L A1 + RA2 L A2) = α (RA1 − RA2)(L A2 − L A1) = α∇ RA∇ L Ax2 (Tsigankov and Koulakov, 2006, p. 106)
    EA = alpha * ( RA(RofL(iL2)) - RA(RofL(iL)) ) * (LA(iL2)-LA(iL)) in Savier 2017 

    Returns:
        np.ndarray: the difference in chemical energy between all pairs of compared axons
    """
    ephA_diff = df[1][ax1] - df[1][ax2] # df[1] is EphA
    efnA_diff = df[6][ax2] - df[6][ax1] # df[6] is efnA
    return alpha * (ephA_diff) * (efnA_diff) 

@njit(nogil=True)
def eChemB(ax1, ax2, df, beta) ->  float:
    """
     function of the chemical 'strength' between the two positions as defined by the target efnB and source EphB 

    E = α (RA1 L A2 + RA2 L A1) − α (RA1 L A1 + RA2 L A2) = α (RA1 − RA2)(L A2 − L A1) = α∇ RA∇ L Ax2 (Tsigankov and Koulakov, 2006, p. 106)
    EA = alpha * ( RA(RofL(iL2)) - RA(RofL(iL)) ) * (LA(iL2)-LA(iL)) in Savier 2017 

    Returns:
        np.ndarray: the difference in chemical energy between all pairs of compared axons
    """
    ephB_diff = df[2][ax1] - df[2][ax2] # df[1] is EphB
    efnB_diff = df[7][ax2] - df[7][ax1] # df[7] is efnB
    return beta * (ephB_diff) * (efnB_diff) # TODO check this and change it to the correct formula for EphB signalling 

@njit(nogil=True)
def eAct(ax1, ax2, df, R, gamma, d) ->  np.ndarray:
    """
    function of the distance between the axons in the target (SC) vs in the source tissue (ret or V1)

    Returns:
        np.ndarray: the difference in activity energy between all pairs of compared axons
    """

    source_dist = np.linalg.norm(df[3:5, ax1] - df[3:5, ax2]) # df[3:5] are the retinal xy coordinates
    cross_correlation = np.exp(-source_dist/R)              # Cij in Koulakov&Tsigankov 2006
    
    sc_dist = np.linalg.norm(df[8:, ax1] - df[8:, ax2]) # r in Koulakov&Tsigankov 2006 -- df[8:] are the collicular xy coordinates
    form_attract = np.exp(-(sc_dist**2) /(2*d**2))          # U in Koulakov&Tsigankov 2006 - overlap of dendritic arbors (Gaussian) -- 'd' is an experimental value [3%of the SC dimensions, see Methods, Eq. (2)]
    return -(gamma/2) * cross_correlation * form_attract

@njit(nogil=True)
def refine_map_iter(df, alpha, beta, gamma, R, d, n_repeats=3E4, deterministic=True, parallel=True) -> np.ndarray:
    """
    refines the collicular map `n_repeats` times, iteratively 
    Returns:
        np.ndarray: the refined map with the residuals of the refinement process (representing how the refinement progressed)
    """
    # Num = len(df[0]) # number of elements to be paired
    # if Num%2: raise ValueError('There must be an even number of neurons to compare')


    for i in range(int(n_repeats)):
        pairs = random_pairs(df[0].shape[0])                           # makes a complete set of random pairs
        total_energy = all_eTotal(pairs, df, alpha, beta, gamma, R, d)                # calcs if eTotal sufficient for swap
        
        norm_eTot = 1 / (1 + np.exp(4 * total_energy))      # normalization of the energy measurements
        if deterministic: 
            comparator =  np.full(len(pairs), 0.5) 
        else: 
            comparator = np.random.random(len(pairs))
        swap = norm_eTot > comparator                   # large Etotal makes a small normalized value i.e., swap would increse map energy

        df = update_df(pairs[swap], df)                     # exchanges the target locations bewteen the pair
    return df
