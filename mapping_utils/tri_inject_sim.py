#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backend_bases import MouseButton
from mapper import Mapper, Tissue, show_grads
sns.set_context('paper')

import warnings
# warnings.filterwarnings('ignore')

from datetime import datetime
start_time = datetime.now()






# Setting up gradients in the Retina, Superior Colliculus, and the Primary Visual Cortex
Num = 250

retina = Tissue(Num, 'Retina')
colliculus = Tissue(Num, 'SC')
cortex = Tissue(Num, 'V1')

# retinal_gradients = retina.make_std_grads(EphA_angle=90, EphB_angle=45)
retinal_gradients = retina.make_std_grads() # can can the axis along which the gradients are established by passing a corresponding angle


ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict, _, _, _, _ = retinal_gradients
# ret_EphAs_dict['EphA4'] *= retina.isl2 *2 # defining a mutant
ret_efnAs_dict['efnA2'] *= retina.isl2 # defining a mutant


# Establish the axon concentrations of Eph and efn to be used for mapping
retina.set_gradients(EphA=ret_EphAs_dict, EphB=ret_EphBs_dict, efnA=ret_efnAs_dict, efnB=ret_efnBs_dict)

# cc_gradients = colliculus.make_std_grads(efnA_angle=45, efnB_angle=0)
cc_gradients = colliculus.make_std_grads(efnA_angle=0)

_, _, _, _, sc_efnAs_dict, sc_efnBs_dict, cort_EphAs_dict, cort_EphBs_dict = cc_gradients
colliculus.set_gradients(efnA=sc_efnAs_dict, efnB=sc_efnBs_dict)
cortex.set_gradients(EphA=cort_EphAs_dict, EphB=cort_EphBs_dict) 





# Set up the Retino-Collicular Map 
rc = Mapper(gamma=100, alpha=220, beta=220, R=0.11, d=3/Num**2)
rc.name = "Retino-Colliculuar Map"
rc.source_name = 'Retina'
rc.target_name = 'Colliculus'
rc.source_x, rc.source_y = 'Nasal-Temporal', 'Dorso-Ventral'
rc.target_y, rc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
random_RCmap = rc.init_random_map(Num)
rc.make_map_df(random_RCmap, retina, colliculus)

# Set Up the Cortico-Collicular Map
cc = Mapper(gamma=200, alpha=220, beta=220, R=0.11, d=0.001)
cc.name = "Cortico-Colliculuar Map"
cc.source_name = 'Cortex'
cc.target_name = 'Colliculus'
cc.source_x, cc.source_y = 'Lateral-Medial', 'Anterior-Posterior'
cc.target_x, cc.target_y = 'Anterior-Posterior', 'Medial-Lateral'
random_CCmap = cc.init_random_map(Num) # hash representing random connections between the source and target
cc.make_map_df(random_CCmap, cortex, colliculus)

fig = show_grads(rc, cc, retina, colliculus, cortex)
fig.show()
#%%

# Refine the Retino-Collicular Map
rc.df = rc.refine_map(n_repeats=1E3).copy() # Dataframe representing the refined Retino-Collicular Map, from the "perspective" of the RGCs
rc.mapped = rc.df[5].astype(int)
rc.mapped_inv = rc.df[5].argsort().astype(int)

print('Map Refined: {}'.format(datetime.now() - start_time))


# Transpose the Retina Ephrins to the SC
RCmap_hash = rc.df[5].astype(int) # a hashmap that represents the Bijective Connections between the Retina and the SC
CCmap_hash = random_CCmap


ret_efns = np.array([[retina.efnA[*xy], retina.efnB[*xy]] for xy in retina.positions])
transposed_efns = ret_efns[RCmap_hash.argsort()].T # efnA and efnB sorted to represent their transposition into the 'target' SC

cc.df[6], cc.df[7] = transposed_efns[0][CCmap_hash], transposed_efns[1][CCmap_hash]

df_CC = cc.refine_map(n_repeats=1E3).copy() 

print('Map Refined: {}'.format(datetime.now() - start_time))
#%%


def si_src_trg_arrs(df, inject=[0.5,0.5], max_diff=0.025):
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1])) # array representing point
    in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0) # all distances to point (L2 Norm)
    
    inj = in_range_src  
    hash_map = df[5].argsort().astype(int)

    # how the injection would appear in the source
    src_arr = np.zeros((Num,Num))
    for c, i in zip(retina.positions, inj): # recreates the 2D plane from coordinate data
        src_arr[*c] = i

    # how the injection would appear in the target 
    trg_arr = np.zeros((Num, Num))
    for c, i in zip(colliculus.positions, inj[hash_map]): # recreates the 2D plane from coordinate data
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
    trg = trg[::-1,::-1].T ################################ --- look out for this wild flip. It gets everything aligned nicely, but it really messes with the axes
    return src, trg


def tri_injection(df, center, r=0.11):
    src0, trg0 = si_src_trg_arrs(df, [center[0] - r, center[1] - r])
    src1, trg1 = si_src_trg_arrs(df, [center[0] + r, center[1] - r])
    src2, trg2 = si_src_trg_arrs(df, [center[0],center[1] + 0.66*r])
    src0 += src2 *0.88
    src1 += src2 *0.88
    trg0 += trg2 *0.88
    trg1 += trg2 *0.88

    src =  np.array((src0, src1, src2)).T
    trg =  np.array((trg0, trg1, trg2)).T
    return src/src.max(), trg/trg.max()



def follow_cursor(event):
    ''' glued to the cursor when on the map '''
    if event.inaxes:
        inj = [event.xdata/Num, event.ydata/Num]
        ax[0].clear()
        ax[1].clear()
        src, trg = tri_injection(refined_map, inj) # makes three points that represent multiple injections
        # src, trg = si_src_trg_arrs(refined_map, inj)
        ax[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
        ax[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
        fig.canvas.draw()
        
def on_click(event): 
    ''' stop doing `binding_id` on left click '''
    if event.button is MouseButton.LEFT:
        # plt.disconnect(binding_id)
        pass


for map in [rc, cc]:
    
    refined_map = map.df

    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    default_inject = [0.5,0.5]

    src, trg = tri_injection(refined_map, default_inject)

    ax[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
    ax[0].set_title(map.source_name)
    ax[0].set_xlabel(map.source_x)
    ax[0].set_ylabel(map.source_y)
    
    ax[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
    ax[1].set_title(map.target_name)
    ax[0].set_xlabel(map.target_x)
    ax[0].set_ylabel(map.target_y)   
    
    fig.suptitle(map.name)
            
    ax[0].axis('off')
    ax[1].axis('off')

    binding_id = plt.connect('motion_notify_event',follow_cursor)
    plt.connect('button_press_event', on_click)
    plt.show()
# %%


