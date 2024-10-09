#%%
import numpy as np
import matplotlib.pyplot as plt
from mapper import Mapper, Tissue

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
start_time = datetime.now()

# Setting up gradients in the Retina, Superior Colliculus, and the Primary Visual Cortex
Num = 250

retina = Tissue(Num, 'Retina')
colliculus = Tissue(Num, 'SC')
cortex = Tissue(Num, 'V1')

retinal_gradients = retina.make_std_grads(EphA_angle=60, EphB_angle=45)
ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict, _, _, _, _ = retinal_gradients
ret_EphAs_dict['EphA4'] *= retina.isl2 *2 # defining a mutant

# Establish the axon concentrations of Eph and efn to be used for mapping
retina.set_gradients(EphA=ret_EphAs_dict, EphB=ret_EphBs_dict, efnA=ret_efnAs_dict, efnB=ret_efnBs_dict)

cc_gradients = colliculus.make_std_grads()
_, _, _, _, sc_efnAs_dict, sc_efnBs_dict, cort_EphAs_dict, cort_EphBs_dict = cc_gradients
colliculus.set_gradients(efnA=sc_efnAs_dict, efnB=sc_efnBs_dict)
cortex.set_gradients(EphA=cort_EphAs_dict, EphB=cort_EphBs_dict) 


fig, axes = plt.subplots(ncols=2, nrows=2)
axs = axes.flat
axs[0].imshow(retina.EphA)
axs[0].set_title('retina.EphA')
axs[1].imshow(retina.EphB)
axs[1].set_title('retina.EphB')
axs[2].imshow(colliculus.efnA)
axs[2].set_title('colliculus.efnA')
axs[3].imshow(colliculus.efnB)
axs[3].set_title('colliculus.efnB')
fig.tight_layout()
plt.show()

#%%
# Set up the Retino-Collicular Map 
rc = Mapper(gamma=100, alpha=220, beta=220, R=0.11, d=3/Num**2)
random_RCmap = rc.init_random_map(Num)
rc.make_map_df(random_RCmap, retina, colliculus)

# Refine the Retino-Collicular Map
rc.df = rc.refine_map(n_repeats=1E3).copy() # Dataframe representing the refined Retino-Collicular Map, from the "perspective" of the RGCs
rc.mapped = rc.df[5].astype(int)
rc.mapped_inv = rc.df[5].argsort().astype(int)

print('Map Refined: {}'.format(datetime.now() - start_time))

# # Set Up the Cortico-Collicular Map
# cc = Mapper(gamma=200, alpha=220, beta=220, R=0.11, d=0.001)
# random_CCmap = cc.init_random_map(Num) # hash representing random connections between the source and target

# # Transpose the Retina Ephrins to the SC
# RCmap_hash = rc.df[5].astype(int) # a hashmap that represents the Bijective Connections between the Retina and the SC
# CCmap_hash = random_CCmap


# cc.make_map_df(random_CCmap, cortex, colliculus)
# ret_efns = np.array([[retina.efnA[*xy], retina.efnB[*xy]] for xy in retina.positions])
# transposed_efns = ret_efns[RCmap_hash.argsort()].T # efnA and efnB sorted to represent their transposition into the 'target' SC

# cc.df[6], cc.df[7] = transposed_efns[0][CCmap_hash], transposed_efns[1][CCmap_hash]

# df_CC = cc.refine_map(n_repeats=1E3).copy()


#%%
# fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10,20))
# ax[0,0].scatter(*refined_map[3:5], c=refined_map[9], cmap='turbo', s=4)
# ax[0,0].set_title('SCy')
# ax[1,0].scatter(*refined_map[3:5], c=refined_map[8], cmap='turbo', s=4)
# ax[1,0].set_title('SCx')
# ax[2,0].scatter(*refined_map[3:5], c=refined_map[6], cmap='turbo', s=4)
# ax[2,0].set_title('efnA')
# ax[3,0].scatter(*refined_map[3:5], c=refined_map[7], cmap='turbo', s=4)
# ax[3,0].set_title('efnB')

# ax[0,1].scatter(*refined_map[8:], c=refined_map[4], cmap='turbo', s=4)
# ax[0,1].set_title('Rety')
# ax[1,1].scatter(*refined_map[8:], c=refined_map[3], cmap='turbo', s=4)
# ax[1,1].set_title('RetX')

# ax[2,1].scatter(*refined_map[8:], c=refined_map[1], cmap='turbo', s=4)
# ax[2,1].set_title('EphA')
# ax[3,1].scatter(*refined_map[8:], c=refined_map[2], cmap='turbo', s=4)
# ax[3,1].set_title('EphB')
""""""
# %%

# fig, ax = plt.subplots(nrows=2, figsize=(5,10))
# ax[0].scatter(*refined_map[8:], c=refined_map[4], cmap='turbo_r', s=4)
# ax[0].set_title('SC per Rety')
# ax[1].scatter(*refined_map[3:5], c=refined_map[6], cmap='turbo', s=4)
# ax[1].set_title('Ret per efnA')

# # id_src, EphA_at_src, EphB_at_src, pos_at_src.T[0], pos_at_src.T[1], RCmap, efnA_at_trg, efnB_at_trg, pos_at_trg.T[0], pos_at_trg.T[1]
# # 0,      1,           2,           3,               4,               5,     6,           7,           8,               9, 
# # %%


# %%
#%%
def source_injection(df, inject=[0.5,0.5]):
    
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1]))
    # how the injection would appear in the source
    
    
    in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0)
    inj = in_range_src
    
    # in_range_trg = np.linalg.norm(df[8:] - injection_arr, axis=0) # this needs to use the inverse map to work TODO
    
    # if site=='source': inj = in_range_src
    # else: inj = in_range_trg
    
    
    src_arr = np.zeros((Num,Num))
    for c, i in zip(retina.positions, inj):
        src_arr[*c] = i

    # how the injection would appear in the target 
    trg_arr = np.zeros((Num, Num))
    for c, i in zip(colliculus.positions, inj[df[5].astype(int)]):
        trg_arr[*c] = i
        
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
    ax[0,0].imshow(src_arr.T<0.05, cmap='Greys_r')
    ax[0,0].set_title('Retina')
    ax[0,1].imshow(trg_arr.T<0.05, cmap='Greys_r')
    ax[0,1].set_title('Superior Colliculus')
    ax[1,0].imshow((src_arr.T-0.04)**-0.1, cmap='Greys_r')
    ax[1,1].imshow((trg_arr.T-0.04)**-0.1, cmap='Greys_r')
    fig.tight_layout()
    return fig

# fig = source_injection(refined_map, [0.5,0.5])


end_time = datetime.now()
print('Total Duration: {}'.format(end_time - start_time))

# %%
# df = refined_map.copy()
# rc = Mapper(df, gamma=0, alpha=220, beta=220, R=0.11, d=0.01)



refined_map = rc.df

def si_src_trg_arrs(df, inject=[0.5,0.5], max_diff=0.025):
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1])) # array representing point
    in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0) # all distances to point (L2 Norm)
    # in_range_src[np.where(in_range_src>max_diff)] = 0
    inj = in_range_src  
    hash_map = rc.mapped_inv

    # how the injection would appear in the source
    src_arr = np.zeros((Num,Num))
    for c, i in zip(retina.positions, inj):
        src_arr[*c] = i

    # how the injection would appear in the target 
    trg_arr = np.zeros((Num, Num))
    for c, i in zip(colliculus.positions, inj[hash_map]):
        trg_arr[*c] = i
        
    src_mask = src_arr>max_diff
    trg_mask = trg_arr>max_diff

    src = src_arr**0.1 *5
    trg = trg_arr**0.1 *5
    
    src = (1 - (src/src.max()) * src_mask) 
    trg = (1 - (trg/trg.max()) * trg_mask)
    # trg = trg[::-1]

    # src = np.tile(src, [3,1,1]).T
    # trg = np.tile(trg, [3,1,1]).T
    return src, trg

import matplotlib.cm as cm      
from matplotlib.backend_bases import MouseButton


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
        src, trg = tri_injection(refined_map, inj)
        # src, trg = si_src_trg_arrs(refined_map, inj)
        ax[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
        ax[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
        fig.canvas.draw()
        
def on_click(event): 
    ''' stop doing `binding_id` on left click '''
    if event.button is MouseButton.LEFT:
        plt.disconnect(binding_id)


# im = plt.imread('../Color-Wheel.jpg')
# imw = np.array(im.copy())
# trunc = imw[450:450+Num, 450:450+Num]
# trunc = np.array(trunc.astype(int)) # TODO -- there's a problem around here where the image is being displayed really weird *************
# # trunc = trunc[:,:,2]
# warped = np.zeros([250,250,3])
# for src, trg in zip(retina.positions, colliculus.positions[rc.mapped_inv]):
#     warped[*trg] = trunc[*src].astype(int)
# # warped = warped[:,::,::-1]

fig, ax = plt.subplots(ncols=2, figsize=(10,5))
default_inject = [0.5,0.5]

src, trg = tri_injection(refined_map, default_inject)

print(src.shape)


print(f'{src.shape = }')
print(f'{trg.shape = }')
ax[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
ax[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)

        
plt.axis('off')
binding_id = plt.connect('motion_notify_event',follow_cursor)
plt.connect('button_press_event', on_click)
plt.show()
# %%

# plt.imshow(warped.astype(int))

# %%
