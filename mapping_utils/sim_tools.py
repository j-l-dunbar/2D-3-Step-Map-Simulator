import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.backend_bases import MouseButton
from mapper import Mapper, Tissue#, make_std_tissues, run_map_sim
sns.set_style('darkgrid')
sns.set_context('paper')

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# def coords_to_img(coords, values, Num):
#     new_img = np.ones((Num, Num))
#     for xy, i in zip(coords, values):
#         new_img[*xy] = i
#     return new_img




def make_std_tissues(sim_params):
    Num = sim_params['Num']
    # Setting up gradients in the Retina, Superior Colliculus, and the Primary Visual Cortex
    retina = Tissue(Num, 'Retina')
    colliculus = Tissue(Num, 'SC')
    cortex = Tissue(Num, 'V1')

    # retinal_gradients = retina.make_std_grads(EphA_angle=90, EphB_angle=45)
    retinal_gradients = retina.make_std_grads() # can can the axis along which the gradients are established by passing a corresponding angle

    # Establish the axon concentrations of Eph and efn to be used for mapping
    retina.EphA_dict, retina.EphB_dict, retina.efnA_dict, retina.efnB_dict, _, _, _, _ = retinal_gradients

    # setting the direction of the efnA gradietns -- as they efnBs are not yet measured, there are conflicting conventions here 
    cc_gradients = colliculus.make_std_grads(efnA_angle=0) 
    _, _, _, _, colliculus.efnA_dict, colliculus.efnB_dict, cortex.EphA_dict, cortex.EphB_dict = cc_gradients
    
    return retina, colliculus, cortex

def run_map_sim(retina, colliculus, cortex, sim_params):
    retina.set_gradients()
    colliculus.set_gradients()
    cortex.set_gradients() 

    # Set up the Retino-Collicular Map 
    rc = Mapper(**sim_params)
    rc.name = "Retino-Collicular Map"
    rc.source_name, rc.target_name = 'Retina', 'Colliculus'
    rc.source_x, rc.source_y = 'Nasal-Temporal', 'Ventro-Dorsal'
    rc.target_y, rc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
    random_RCmap = rc.init_random_map()
    rc.make_map_df(random_RCmap, retina, colliculus)

    # Set Up the Cortico-Collicular Map
    cc = Mapper(**sim_params)
    cc.name = "Cortico-Collicular Map"
    cc.source_name, cc.target_name = 'Cortex', 'Colliculus'
    cc.source_x, cc.source_y = 'Lateral-Medial', 'Anterior-Posterior'
    cc.target_y, cc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
    random_CCmap = cc.init_random_map() # hash representing random connections between the source and target
    cc.make_map_df(random_CCmap, cortex, colliculus)
    cc.hash_map = cc.df[5].astype(int) # random hashmap to be used to set up retinal gradient transposition
    cc.hash_map_inv = cc.df[5].argsort().astype(int) # random hashmap to be used to set up retinal gradient transposition

    rc_fig_grads = rc.df_show_grads()
    fig = rc_fig_grads

    
    
    x_names = [rc.source_x, rc.target_x, rc.source_x, rc.target_x]
    y_names = [rc.source_y, rc.target_y, rc.source_y, rc.target_y]
    ax_titles = ['EphA', 'efnA', 'EphB', 'efnB']
    for ax, x, y, title in zip(fig.axes, x_names, y_names, ax_titles):    
        ax.set_title(title, size=22)
        ax.set_xlabel(x, size=18)
        ax.set_ylabel(y, size=18)
    fig.suptitle("Retino-Collicular Gradients")
    fig.tight_layout()
    rc_fig_grads = fig

        
    #################################################################
    ############## Refine the Retino-Collicular Map #################
    #################################################################

    # Refine the Retino-Collicular Map
    start_time = datetime.now()
    rc.df = rc.refine_map(n_repeats=1E3).copy() # Dataframe representing the refined Retino-Collicular Map, from the "perspective" of the RGCs
    rc.hash_map = rc.df[5].astype(int) # a hashmap that represents the Bijective Connections between the Retina and the SC
    rc.hash_map_inv = rc.df[5].argsort().astype(int)
    print('Map Refined: {}'.format(datetime.now() - start_time))
    lap_time = datetime.now()

    #################################################################
    ############ Transpose the Retina Ephrins to the SC #############
    #################################################################

    # efnA and efnB sorted to represent their transposition into the 'target' SC
    source_efns = np.array([[retina.efnA[*xy], retina.efnB[*xy]] for xy in retina.positions]).T
    if sim_params['complex_transpose']: # this is a more complicated mechanism for the 'projection' of reitnal ephrins to the SC
        print('Eph removes collicular efn, proportionally')
        modified_efns = rc.df[6:8] - rc.df[1:3]*0.5 # col_efns_perRet - ret_Ephs --> simulating the partial destruction of efns by retinal Eph
        modified_efns += source_efns # transposed retinal ephrins are added to the remaining collicular efn
        efns_in_sc = modified_efns[:,rc.hash_map_inv] # the array is sorted wrt the target 
    else: # mechanism from the original 3 Step Map Alighment Model (Savier, 2017)
        print('Simple Replacement Mechanism')
        efns_in_sc = source_efns[:,rc.hash_map_inv]
        
    efns_to_cortex = efns_in_sc[:,cc.hash_map] # efnAs to be used for CC map refinement, ordered per the source (cortex)
    cc.df[6:8] = efns_to_cortex

    cc_fig_grads = cc.df_show_grads()
    fig = cc_fig_grads
    x_names = [cc.source_x, cc.target_x, cc.source_x, cc.target_x]
    y_names = [cc.source_y, cc.target_y, cc.source_y, cc.target_y]
    ax_titles = ['EphA', 'efnA', 'EphB', 'efnB']
    for ax, x, y, title in zip(fig.axes, x_names, y_names, ax_titles):    
        ax.set_title(title, size=22)
        ax.set_xlabel(x, size=18)
        ax.set_ylabel(y, size=18)
    fig.suptitle("Cortico-Collicular Gradients")
    fig.tight_layout()
    cc_fig_grads = fig
    
    #################################################################
    ############## Refine the Cortico-Collicular Map ################
    #################################################################

    cc.df = cc.refine_map(n_repeats=1E3).copy()
    cc.hash_map = cc.df[5].astype(int)
    cc.hash_map_inv = cc.df[5].argsort()
    print('Map Refined: {}'.format(datetime.now() - lap_time))

    return rc, cc, rc_fig_grads, cc_fig_grads



def sim_efnA_ki(strength, sim_params):
    'Simulates an Isl2-mediated efnA knock in of a specified strength'
    # initialize the gradients to be used for mapping
    retina, colliculus, cortex = make_std_tissues(sim_params)
    mutations, retina.efnA_dict =  retina.make_isl2_ki('efnA', strength, retina.efnA_dict)
    # run the 3 Step Map Alignment Model
    rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)
    
    return mutations, retina, colliculus, rc, cc, rc_fig_grads, cc_fig_grads

def sim_efnA_ko(target, sim_params):
    'Simulates an Isl2-mediated efnA knock out of a specified strength'
    # initialize the gradients to be used for mapping
    retina, colliculus, cortex = make_std_tissues(sim_params)
    mutations, retina.efnA_dict =  retina.make_isl2_ko(target, retina.efnA_dict)
    # run the 3 Step Map Alignment Model
    rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)
    
    return mutations, retina, colliculus, rc, cc, rc_fig_grads, cc_fig_grads

def sim_EphA_ki(strength, sim_params):
    'Simulates an Isl2-mediated EphA knock in of a specified strength'
    # initialize the gradients to be used for mapping
    retina, colliculus, cortex = make_std_tissues(sim_params)
    mutations, retina.EphA_dict =  retina.make_isl2_ki('EphA', strength, retina.EphA_dict)
    # run the 3 Step Map Alignment Model
    rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)
    
    return mutations, retina, colliculus, rc, cc, rc_fig_grads, cc_fig_grads

def sim_EphA_ko(target, sim_params):
    'Simulates an Isl2-mediated EphA knock out of a specified strength'
    # initialize the gradients to be used for mapping
    retina, colliculus, cortex = make_std_tissues(sim_params)
    mutations, retina.EphA_dict =  retina.make_isl2_ko(target, retina.EphA_dict)
    # run the 3 Step Map Alignment Model
    rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)
    
    return mutations, retina, colliculus, rc, cc, rc_fig_grads, cc_fig_grads

def save_grad_pics(mutations, rc_fig_grads, cc_fig_grads, dpi=300):
    grads_title = f"{'-'.join(mutations)}"
    fname_grads = grads_title.replace('/','').replace(' ', '')
    rc_fig_grads.savefig(fname_grads + '_rc.png', dpi=dpi)
    cc_fig_grads.savefig(fname_grads + '_cc.png', dpi=dpi)
