#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import seaborn as sns
from fractions import Fraction
from matplotlib.backend_bases import MouseButton
from mapper import Mapper, Tissue, show_grads
sns.set_style('darkgrid')
sns.set_context('paper')

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime


def fractional_axes(axes, Num, color='k', numticks=9):
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

# Setting up gradients in the Retina, Superior Colliculus, and the Primary Visual Cortex
Num = 250
sim_params = {'gamma':100, 'alpha':220, 'beta':220, 'R':0.11, 'd':3/Num**2}
show_grads_bool = True

retina = Tissue(Num, 'Retina')
colliculus = Tissue(Num, 'SC')
cortex = Tissue(Num, 'V1')

# retinal_gradients = retina.make_std_grads(EphA_angle=90, EphB_angle=45)
retinal_gradients = retina.make_std_grads() # can can the axis along which the gradients are established by passing a corresponding angle


# Establish the axon concentrations of Eph and efn to be used for mapping
ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict, _, _, _, _ = retinal_gradients
figure_title="Wildtype"
# figure_title, ret_efnAs_dict = retina.make_isl2_ko('efnA2', ret_efnAs_dict)
figure_title, ret_EphAs_dict = retina.make_isl2_ki('EphA3', 1, ret_EphAs_dict)
# ret_EphAs_dict['EphA4'] *= retina.isl2 # defining a mutant
# ret_efnAs_dict['efnA2'] *= retina.isl2 # defining a mutant

retina.set_gradients(EphA=ret_EphAs_dict, EphB=ret_EphBs_dict, efnA=ret_efnAs_dict, efnB=ret_efnBs_dict)
cc_gradients = colliculus.make_std_grads(efnA_angle=0)
_, _, _, _, sc_efnAs_dict, sc_efnBs_dict, cort_EphAs_dict, cort_EphBs_dict = cc_gradients
colliculus.set_gradients(efnA=sc_efnAs_dict, efnB=sc_efnBs_dict)
cortex.set_gradients(EphA=cort_EphAs_dict, EphB=cort_EphBs_dict) 

# Set up the Retino-Collicular Map 
rc = Mapper(**sim_params)
rc.name = "Retino-Colliculuar Map"
rc.source_name, rc.target_name = 'Retina', 'Colliculus'
rc.source_x, rc.source_y = 'Nasal-Temporal', 'Dorso-Ventral'
rc.target_y, rc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
random_RCmap = rc.init_random_map(Num)
rc.make_map_df(random_RCmap, retina, colliculus)

# Set Up the Cortico-Collicular Map
cc = Mapper(**sim_params)
cc.name = "Cortico-Colliculuar Map"
cc.source_name, cc.target_name = 'Cortex', 'Colliculus'
cc.source_x, cc.source_y = 'Lateral-Medial', 'Anterior-Posterior'
cc.target_y, cc.target_x = 'Anterior-Posterior', 'Medial-Lateral'
random_CCmap = cc.init_random_map(Num) # hash representing random connections between the source and target
cc.make_map_df(random_CCmap, cortex, colliculus)



if show_grads_bool:
    fig = show_grads(rc, cc, retina, colliculus, cortex)
    fractional_axes(fig.axes, Num, 'w', numticks=5)
    plt.show()

#%%

start_time = datetime.now()
# Refine the Retino-Collicular Map
rc.df = rc.refine_map(n_repeats=1E3).copy() # Dataframe representing the refined Retino-Collicular Map, from the "perspective" of the RGCs
rc.mapped = rc.df[5].astype(int)
rc.mapped_inv = rc.df[5].argsort().astype(int)
print('Map Refined: {}'.format(datetime.now() - start_time))


lap_time = datetime.now()
# Transpose the Retina Ephrins to the SC
RCmap_hash = rc.df[5].astype(int) # a hashmap that represents the Bijective Connections between the Retina and the SC
CCmap_hash = random_CCmap


ret_efns = np.array([[retina.efnA[*xy], retina.efnB[*xy]] for xy in retina.positions])
transposed_efns = ret_efns[RCmap_hash.argsort()].T # efnA and efnB sorted to represent their transposition into the 'target' SC

cc.df[6], cc.df[7] = transposed_efns[0][CCmap_hash], transposed_efns[1][CCmap_hash]

df_CC = cc.refine_map(n_repeats=1E3).copy() 

print('Map Refined: {}'.format(datetime.now() - lap_time))
#%%


def si_src_trg_arrs(df, inject=[0.5,0.5], max_diff=0.025, anterograde=True):
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1])) # array representing point
    # in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0) # all distances to point (L2 Norm)
    in_range_src = np.linalg.norm((df[8:] - injection_arr), axis=0) # all distances to point (L2 Norm)
    
    inj = in_range_src  
    hash_map_trg = df[5].argsort().astype(int)
    hash_map_src = df[5].astype(int)
    
    if anterograde: 
        hash_map_src = np.ones_like(hash_map_src)
    else:
        hash_map_trg = np.ones_like(hash_map_trg)

    # how the injection would appear in the source
    src_arr = np.zeros((Num,Num))
    for c, i in zip(retina.positions, inj[hash_map_src]): # recreates the 2D plane from coordinate data
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
    trg = trg[::-1,::-1].T ################################ --- look out for this wild flip. It gets everything aligned nicely, but it really messes with the axes
    return src, trg


def tri_injection(df, center, r=1/8):
    src0, trg0 = si_src_trg_arrs(df, [center[0] - r, center[1] - 2*r])
    src1, trg1 = si_src_trg_arrs(df, [center[0] + r, center[1] - 2*r])
    src2, trg2 = si_src_trg_arrs(df, [center[0]    , center[1]])
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
        # fig.canvas.flush_events()
        
        src_rc, trg_rc = tri_injection(col_map.df, inj)

        ax[0].imshow(src_rc, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
        ax[1].imshow(trg_rc, cmap='Greys_r', origin='lower', vmax=1, vmin=0)

        fig.canvas.draw()
        
def on_click(event): 
    ''' stop doing `binding_id` on left click '''
    if event.button is MouseButton.LEFT:
        # plt.disconnect(binding_id)
        pass



# for col_map in [cc]:
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
    
    fractional_axes(ax, Num, 'k')
    

    

    fig.suptitle(f'{figure_title} - {col_map.name} - Anterograde', size=20)
            
    # ax[0].axis('off')
    # ax[1].axis('off')

    binding_id = plt.connect('motion_notify_event',follow_cursor)
    plt.connect('button_press_event', on_click)
    plt.show()

# %%
print('Total Time: {}'.format(datetime.now() - lap_time))

