#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mapper import Mapper, Tissue

from datetime import datetime
start_time = datetime.now()




Num = 50
retina = Tissue(Num)
x = retina.grid_fract

# There's a problem with the gradients as inserted
ret_EphBs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    'EphA4': 0.04 * np.exp(x[1]) + 0.939, 
    'EphA5': 0.515 * np.exp(x[1]) + 0.1232, 
    'EphA6': 0.572 * np.exp(x[1]) + 0.03
}

ret_EphAs_dict = { # Adult Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    'EphA4': 0.04 * np.exp(x[0]) + 0.939 * (retina.isl2), 
    'EphA5': 0.515 * np.exp(x[0]) + 0.123, 
    'EphA6': 0.572 * np.exp(x[0]) + 0.03
}

ret_efnBs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    # 'efnA2': (-0.066 * np.exp( -x) +1.045) * 0.11, # only 11% of the efnA5 puncta were counted for efnA2
    'efnA2': (-0.066 * np.exp( -x[1]) +1.045) * 0.11, 
    'efnA3': (0.232 * np.exp(-x[1]) + 0.852)  * 0.22, 
    'efnA5': (1.356 * np.exp(-x[1]) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
}

ret_efnAs_dict = { # P0  Measurement - assumed exponential, curves estimated by kernel densities with a bandwidth of 0.1
    # 'efnA2': (-0.066 * np.exp( -x) +1.045) * 0.11, # only 11% of the efnA5 puncta were counted for efnA2
    'efnA2': (0.066 * np.exp( -x[0]) +1.045) * 0.11, 
    'efnA3': (0.232 * np.exp(-x[0]) + 0.852)  * 0.22, 
    'efnA5': (1.356 * np.exp(-x[0]) + 0.147) * 0.5,  # some guesses as to the final contribution to the summed ephrin gradients
}

sc_efnAs_dict = {
    # 'efnA5': -2.365*x**3 + 2.944*x**2 + 0.325*x + 0.454, # polynomial - should arguably use this over the exponential, as even the corrected efnA5 measurement has a fall-off at the posterior-most SC
    'efnA5': 0.646 * np.exp(x[1]) - 0.106, # exponential
    'efnA3': -0.052 * np.exp(x[1]) + 1.008, # exponential
    'efnA2': -0.124*x[1]**3 - 0.896*x[1]**2 + 1.25*x[1] + 0.708, # polynomial
    # 'theoretical': (np.exp((np.arange(Num) - Num) / Num) - np.exp((-np.arange(Num) - Num) / Num))
} # JD Measured

cort_EphAs_dict = {
    'theoretical': (np.exp((np.arange(Num) - Num) / Num) 
                    - np.exp((-np.arange(Num) - Num) / Num))
} # from Savier et al 2017

retina.set_gradients(ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict)

colliculus = Tissue(Num)
colliculus.set_gradients(ret_EphAs_dict, ret_EphBs_dict, ret_efnAs_dict, ret_efnBs_dict)

print('Gradients Set. Time Elapsed: {}'.format(datetime.now() - start_time))
# RCmap = np.random.permutation(colliculus.positions.size[0])
# RCmap[0]



def make_map_df(RCmap, retina, colliculus):
    
    id_src = np.arange(retina.positions.shape[0])
    EphA_at_src = np.array([retina.EphA[*x] for x in retina.positions])
    EphB_at_src = np.array([retina.EphB[*x] for x in retina.positions])
    pos_at_src = np.array([retina.grid_fract.T[*x] for x in retina.positions])
    
    RCmap = RCmap
    efnA_at_trg = np.array([colliculus.efnA[*x] for x in colliculus.positions[RCmap]])
    efnB_at_trg = np.array([colliculus.efnB[*x] for x in colliculus.positions[RCmap]])
    pos_at_trg = np.array([colliculus.grid_fract.T[*x] for x in colliculus.positions[RCmap]])
    
    return np.vstack((id_src, EphA_at_src, EphB_at_src, pos_at_src.T[0], pos_at_src.T[1], RCmap, efnA_at_trg, efnB_at_trg, pos_at_trg.T[0], pos_at_trg.T[1]))

df = make_map_df(np.random.permutation(Num**2), retina, colliculus)
    

# id_src, EphA_at_src, EphB_at_src, pos_at_src.T[0], pos_at_src.T[1], RCmap, efnA_at_trg, efnB_at_trg, pos_at_trg.T[0], pos_at_trg.T[1]
# 0,      1,           2,           3,               4,               5,     6,           7,           8,               9, 

#%%



rc = Mapper(df, gamma=200, alpha=220, beta=220, R=0.11, d=0.03)
refined_map = rc.refine_map(n_repeats=2E3)

print('Map Refined. Time Elapsed: {}'.format(datetime.now() - start_time))
print(f'{refined_map.shape = }')

#%%
# RCmap = refined_map[5]

# efnA_at_src = np.array([retina.efnA[*x] for x in retina.positions])
# efnB_at_src = np.array([retina.efnB[*x] for x in retina.positions])
# efnA_transp = efnA_at_src[RCmap.argsort()]
# efnB_transp = efnB_at_src[RCmap.argsort()]

# df2 = make_map_df(np.random.permutation(Num**2), retina, colliculus)

# df2[6], df2[7] = efnA_transp[df2[5].astype(int)], efnB_transp[df2[5].astype(int)]

# cc = Mapper(df2, gamma=200, alpha=120, beta=120, R=0.11, d=0.03)
# #%%
# refined_map = cc.refine_map()


#%%
fig, ax = plt.subplots(nrows=4, figsize=(5,20))
ax[0].scatter(*refined_map[3:5], c=refined_map[9], cmap='turbo', s=4)
ax[0].set_title('SCy')
ax[1].scatter(*refined_map[3:5], c=refined_map[8], cmap='turbo', s=4)
ax[1].set_title('SCx')
ax[2].scatter(*refined_map[3:5], c=refined_map[7], cmap='turbo', s=4)
ax[2].set_title('efnB')
ax[3].scatter(*refined_map[3:5], c=refined_map[6], cmap='turbo', s=4)
ax[3].set_title('efnA')
#%%
fig, ax = plt.subplots(nrows=4, figsize=(5,20))
ax[0].scatter(*refined_map[8:], c=refined_map[1], cmap='turbo', s=4)
ax[0].set_title('EphA')
ax[1].scatter(*refined_map[8:], c=refined_map[2], cmap='turbo', s=4)
ax[1].set_title('EphB')
ax[2].scatter(*refined_map[8:], c=refined_map[3], cmap='turbo', s=4)
ax[2].set_title('RetX')
ax[3].scatter(*refined_map[8:], c=refined_map[4], cmap='turbo', s=4)
ax[3].set_title('Rety')
# %%

# fig, ax = plt.subplots(nrows=2, figsize=(5,10))
# ax[0].scatter(*refined_map[8:], c=refined_map[4], cmap='turbo_r', s=4)
# ax[0].set_title('SC per Rety')
# ax[1].scatter(*refined_map[3:5], c=refined_map[6], cmap='turbo', s=4)
# ax[1].set_title('Ret per efnA')

# # id_src, EphA_at_src, EphB_at_src, pos_at_src.T[0], pos_at_src.T[1], RCmap, efnA_at_trg, efnB_at_trg, pos_at_trg.T[0], pos_at_trg.T[1]
# # 0,      1,           2,           3,               4,               5,     6,           7,           8,               9, 
# # %%
# injection_mask = np.logical_and(np.logical_and(refined_map[3]>0.40, refined_map[3]<0.50), np.logical_and(refined_map[4]>0.40, refined_map[4]<0.50))
# fig, ax = plt.subplots(nrows=2, figsize=(5,10))
# ax[0].scatter(*refined_map[3:5], c=injection_mask, cmap='turbo', s=7)
# ax[1].scatter(*refined_map[8:], c=injection_mask, cmap='turbo', s=7)
# ax[1].set_title('')
# # %%
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
    for c, i in zip(colliculus.positions, inj[refined_map[5].astype(int)]):
        trg_arr[*c] = i
        
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
    ax[0,0].imshow(src_arr.T<0.05, cmap='turbo')
    ax[0,0].set_title('Retina')
    ax[0,1].imshow(trg_arr.T<0.05, cmap='turbo')
    ax[0,1].set_title('Superior Colliculus')
    ax[1,0].imshow(src_arr.T**0.001, cmap='turbo_r')
    ax[1,1].imshow(trg_arr.T**0.001, cmap='turbo_r')
    fig.tight_layout()
    return fig
#%%
def source_injection2(df, inject=[0.5,0.5]):
    injection_arr = np.vstack((np.ones_like(df[3])*inject[0], np.ones_like(df[4])*inject[1])) # array representing point
    in_range_src = np.linalg.norm((df[3:5] - injection_arr), axis=0) # all distances to point (L2 Norm)
    inj = in_range_src  
    hash_map = refined_map[5].astype(int)

    # how the injection would appear in the source
    src_arr = np.zeros((Num,Num))
    for c, i in zip(retina.positions, inj):
        src_arr[*c] = i

    # how the injection would appear in the target 
    trg_arr = np.zeros((Num, Num))
    for c, i in zip(colliculus.positions, inj[hash_map]):
        trg_arr[*c] = i
        
    return src_arr.T**0.001#, trg_arr.T**0.001

fig = source_injection(refined_map, [0.5,0.5])


end_time = datetime.now()
print('Total Duration: {}'.format(end_time - start_time))


# %%

fig1 = source_injection(refined_map, [0.2,0.22])

plt.show()
# %%

import matplotlib.cm as cm      
from matplotlib.backend_bases import MouseButton

fig = plt.imshow(source_injection2(refined_map, [0.5,0.5]), cm.gray_r)

def onclick(event):
    if event.button is MouseButton.LEFT:
         inj = [event.xdata/Num, event.ydata/Num]
    # clear frame
    fig.set_data(source_injection2(refined_map, inj))
    print(f'Clicked at: {inj}')


plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.connect('button_press_event',onclick)


plt.show()
# %%

