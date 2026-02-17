#%%
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.backend_bases import MouseButton
from mapper import Mapper, Tissue
from sim_tools import make_std_tissues, run_map_sim
sns.set_style('darkgrid')
sns.set_context('paper')

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
start_time = datetime.now()



Num = 250

sim_params = {'gamma':100, 
              'alpha':220, 
              'beta':220, 
              'R':0.11, 
              'd':3/Num**2, 
              'Num':Num, 
              'show_grads_bool':False, 
              'complex_transpose':False,
              }

# initialize the gradients to be used for mapping
retina, colliculus, cortex = make_std_tissues(sim_params)
mutations=[]

# define any mutants
# target_name = 'efnA5'
# mutations.append(target_name + 'ko/ko')
# retina.efnA_dict[target_name] *= 0
# colliculus.efnA_dict[target_name] *= 0

# mutant_part, retina.EphA_dict =  retina.make_isl2_ki('EphA Large', 5, retina.EphA_dict)
# mutant_part, retina.EphB_dict =  retina.make_isl2_ki('EphB Large', 0.5, retina.EphB_dict)
# mutant_part, retina.efnA_dict =  retina.make_isl2_ki('efnA3ki', 5, retina.efnA_dict)
# mutant_part, retina.efnB_dict =  retina.make_isl2_ki('efnB Large',3, retina.efnB_dict)

# mutant_title.append('_' + mutant_part)


# run the 3 Step Map Alignment Model
rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)


if not mutations: mutations = ['Wildtype']


def si_src_trg_arrs(df, inject=[0.5,0.5], max_diff=0.025, retro=False):
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1])) # array representing the injectios point
    
    if retro: in_range_src = np.linalg.norm((df[8:] - injection_arr), axis=0) # all distances to point in target (L2 Norm) 
    else: in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0) # all distances to point in source (L2 Norm)
    
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
    src = (1 - (src/src.max()) * src_mask).T
    trg = (1 - (trg/trg.max()) * trg_mask)
    
    trg = trg[::-1] ################################ --- the x axis is flipped to maintain alighment between the diagrams (which are actually mirror images)
    return src, trg


def tri_injection(df, center, r=1/8, retro=False):
    coords1 = [center[0] - r, center[1] - 2*r] 
    coords2 = [center[0] + r, center[1] - 2*r] 
    coords3 = [center[0]    , center[1]      ]

    if retro:
        coords1 = [center[0] + 2*r, center[1] + r]
        coords2 = [center[0] + 2*r, center[1] - r]
        coords3 = coords3
    
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
        inj = [event.ydata/Num, event.xdata/Num]
        if retro: inj = [1-event.xdata/Num, event.ydata/Num]
        
        axes[0].clear()
        axes[1].clear()
        # fig.canvas.flush_events()
        
        src, trg = tri_injection(col_map.df, inj, retro=retro)

        # if retro: # makes the retrograde injections the photo negative
        #     src, trg = 1-src, 1-trg

        axes[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0) # image of staned axons at the source
        axes[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0) # image of staned axons at the target

        set_axis_labels(col_map, axes)
        fig.canvas.draw()
        
def on_click(event): 
    ''' stop doing `binding_id` on left click '''
    if event.button is MouseButton.LEFT:
        # ant_ret ^= 1
        pass

def set_axis_labels(col_map, axes):
    axes[0].set_title(col_map.source_name, size=15)
    axes[0].set_xlabel(col_map.source_x, size=12)
    axes[0].set_ylabel(col_map.source_y, size=12)

    axes[1].set_title(col_map.target_name, size=15)
    axes[1].set_xlabel(col_map.target_y, size=12)
    axes[1].set_ylabel(col_map.target_x, size=12)  
    rc.fractional_axes(axes, Num, 'k')


for col_map in [rc, cc]:
    fig, axes = plt.subplots(ncols=2, figsize=(10,5))
    axes = axes.flat
    default_inject = [0.5,0.5+1/8]

    src, trg = tri_injection(col_map.df, default_inject)

    axes[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
    axes[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)

    set_axis_labels(col_map, axes)

    fig.suptitle(f'{"-".join(mutations)} - {col_map.name} - Anterograde', size=20)
            
    binding_id = plt.connect('motion_notify_event',follow_cursor)
    plt.connect('button_press_event', on_click)
    plt.show()


print('Total Time: {}'.format(datetime.now() - start_time))


#%%