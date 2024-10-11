#%%
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
start_time = datetime.now()




Num = 100
sim_params = {'gamma':100, 'alpha':220, 'beta':220, 'R':0.11, 'd':3/Num**2, 'Num':Num}

show_grads_bool = True
complex_transpose = False


def make_std_tissues(sim_params):
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
    rc.name = "Retino-Colliculuar Map"
    rc.source_name, rc.target_name = 'Retina', 'Colliculus'
    rc.source_x, rc.source_y = 'Nasal-Temporal', 'Dorso-Ventral'
    rc.target_y, rc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
    random_RCmap = rc.init_random_map()
    rc.make_map_df(random_RCmap, retina, colliculus)

    # Set Up the Cortico-Collicular Map
    cc = Mapper(**sim_params)
    cc.name = "Cortico-Colliculuar Map"
    cc.source_name, cc.target_name = 'Cortex', 'Colliculus'
    cc.source_x, cc.source_y = 'Lateral-Medial', 'Anterior-Posterior'
    cc.target_y, cc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
    random_CCmap = cc.init_random_map() # hash representing random connections between the source and target
    cc.make_map_df(random_CCmap, cortex, colliculus)
    cc.hash_map = cc.df[5].astype(int) # random hashmap to be used to set up retinal gradient transposition
    cc.hash_map_inv = cc.df[5].argsort().astype(int) # random hashmap to be used to set up retinal gradient transposition

    rc_fig_grads = rc.df_show_grads()
    if show_grads_bool:
        plt.show()
        
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
    transposed_efns = np.array([[retina.efnA[*xy], retina.efnB[*xy]] for xy in retina.positions])[rc.hash_map_inv].T
    transposed_Ephs = np.array([[retina.EphA[*xy], retina.EphB[*xy]] for xy in retina.positions])[rc.hash_map_inv].T

    # collicular efns are combined wiht retinal Ephs and ephrins
    efns_in_sc = cc.df[6:8][:,cc.hash_map_inv] # the efns in the SC
    
    if complex_transpose:
        efns_in_sc -= transposed_Ephs # the Eph/ephrin complexes are removed
        efns_in_sc *= efns_in_sc>0 # negative values not considered
        efns_in_sc += transposed_efns
    else:
        efns_in_sc = transposed_efns

    efns_to_cortex = efns_in_sc[:,cc.hash_map] # efnAs to be used for CC map refinement
    cc.df[6:8] = efns_to_cortex

    cc_fig_grads = cc.df_show_grads()
    if show_grads_bool:
        plt.show()
        
    #################################################################
    ############## Refine the Cortico-Collicular Map ################
    #################################################################
    cc.refine_map(n_repeats=1E3)
    print('Map Refined: {}'.format(datetime.now() - lap_time))

    return rc, cc, rc_fig_grads, cc_fig_grads



retina, colliculus, cortex = make_std_tissues(sim_params)
mutant_title, retina.EphA_dict =  retina.make_isl2_ki('EphA Large',3, retina.EphA_dict)
rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)









def si_src_trg_arrs(df, inject=[0.5,0.5], max_diff=0.025, retro=False):
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1])) # array representing the injectios point
    
    if retro: 
        in_range_src = np.linalg.norm((df[8:] - injection_arr), axis=0) # all distances to point in target (L2 Norm) 
    else: 
        in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0) # all distances to point in source (L2 Norm)
    
    inj = in_range_src  
    hash_map_trg = df[5].argsort().astype(int)  # to convert the infromation from retinal space to collicular space

    # how the injection would appear in the source
    src_arr = np.zeros((Num,Num))
    for c, i in zip(retina.positions, inj): # recreates the 2D plane from coordinate data
        src_arr[*c] = i

    # how the injection would appear in the target 
    trg_arr = np.zeros((Num, Num))
    for c, i in zip(colliculus.positions, inj[hash_map_trg]): # recreates the 2D plane from coordinate data
        trg_arr[*c] = i
    
    # adds a defined spot in the middle of the decaying 'glow' gradient    
    src_mask = src_arr>max_diff
    trg_mask = trg_arr>max_diff

    # gets the 'glow' gradient just right
    src = src_arr**0.1 *5 
    trg = trg_arr**0.1 *5
    
    # makes the spots bright instead of dark
    src = (1 - (src/src.max()) * src_mask)
    trg = (1 - (trg/trg.max()) * trg_mask)
    
    trg = trg[::-1,::-1].T ################################ --- look out for this wild flip. It gets everything aligned nicely, but it undoes the reflection seen in the actual system
    return src, trg


def tri_injection(df, center, r=1/8, retro=False):
    coords1 = [center[0] - r, center[1] - 2*r] 
    coords2 = [center[0] + r, center[1] - 2*r] 
    coords3 = [center[0]    , center[1]      ]

    if retro:
        coords1 = [center[0] - r, center[1] + 2*r] [::-1]
        coords2 = [center[0] + r, center[1] + 2*r] [::-1]
        coords3 = coords3[::-1]
    
    src0, trg0 = si_src_trg_arrs(df, coords1, retro=retro)
    src1, trg1 = si_src_trg_arrs(df, coords2, retro=retro)
    src2, trg2 = si_src_trg_arrs(df, coords3, retro=retro)
    
    # make the blue channel white
    src0 += src2 *0.88
    src1 += src2 *0.88
    trg0 += trg2 *0.88
    trg1 += trg2 *0.88
    
    # displayable image
    src =  np.array((src0, src1, src2)).T
    trg =  np.array((trg0, trg1, trg2)).T
    
    return src/src.max(), trg/trg.max()


def follow_cursor(event):
    ''' glued to the cursor when on the map '''
    if event.inaxes:
        retro = bool(event.modifiers)
        inj = [event.xdata/Num, event.ydata/Num]
        if retro: inj = [1-event.xdata/Num, 1-event.ydata/Num]
        
        ax[0].clear()
        ax[1].clear()
        # fig.canvas.flush_events()
        
        src, trg = tri_injection(col_map.df, inj, retro=retro)

        # if retro: # makes the retrograde injections the photo negative
        #     src, trg = 1-src, 1-trg

        ax[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0) # image of staned axons at the source
        ax[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0) # image of staned axons at the target

        fig.canvas.draw()
        
def on_click(event): 
    ''' stop doing `binding_id` on left click '''
    if event.button is MouseButton.LEFT:
        # ant_ret ^= 1
        pass


for col_map in [rc, cc]:
    fig, axes = plt.subplots(ncols=2, figsize=(10,5))
    ax = axes.flat
    default_inject = [0.5,0.5+1/8]

    src, trg = tri_injection(col_map.df, default_inject)

    ax[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
    ax[0].set_title(col_map.source_name, size=15)
    ax[0].set_xlabel(col_map.source_y, size=12)
    ax[0].set_ylabel(col_map.source_x, size=12)

    ax[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
    ax[1].set_title(col_map.target_name, size=15)
    ax[1].set_xlabel(col_map.target_x, size=12)
    ax[1].set_ylabel(col_map.target_y, size=12)  
    
    rc.fractional_axes(ax, Num, 'k')

    fig.suptitle(f'{mutant_title} - {col_map.name} - Anterograde', size=20)
            
    binding_id = plt.connect('motion_notify_event',follow_cursor)
    plt.connect('button_press_event', on_click)
    plt.show()


print('Total Time: {}'.format(datetime.now() - start_time))

#%%