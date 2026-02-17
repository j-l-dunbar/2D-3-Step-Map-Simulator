#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import seaborn as sns
from datetime import datetime
import warnings
import pickle

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
sns.set_context('paper')
start_time = datetime.now()


Num = 350
show_grads_bool = False
save_grads_bool = True
complex_transpose = False

sim_params = {'gamma':100, 'alpha':220, 'beta':220, 'R':0.11, 'd':3/Num**2, 'Num':Num, 'show_grads_bool':show_grads_bool, 'complex_transpose':False}

fname = r'..\pickled_sims\efnA_KI_dicts.pkl'
with open(fname, 'rb') as f:
    mutant_frames = pickle.load(f)
mutations, retina, colliculus, rc, cc = list(mutant_frames.values())[0]

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


def tri_injection(df, center, r=3/32, retro=False):
    coords1 = [center[0] - r* np.sqrt(2), center[1] - r* np.sqrt(2)] # Red and Green
    coords2 = [center[0] - r* np.sqrt(2), center[1] + r* np.sqrt(2)] 
    coords3 = [center[0] +r , center[1] ] # White
    
    src0, trg0 = si_src_trg_arrs(df, coords1, retro=retro)
    src1, trg1 = si_src_trg_arrs(df, coords2, retro=retro)
    src2, trg2 = si_src_trg_arrs(df, coords3, retro=retro)
    
    # make the blue channel white
    c_tint = 0.88
    src0 += src2 *c_tint
    src1 += src2 *c_tint
    trg0 += trg2 *c_tint
    trg1 += trg2 *c_tint
    
    # displayable image
    src =  np.array((src0, src1, src2)).T
    trg =  np.array((trg0, trg1, trg2)).T
    
    return src/src.max(), trg/trg.max()

def update_figure(phi):
        inj = circle(phi)
        axes[0].clear()
        axes[1].clear()
        
        src, trg = tri_injection(col_map.df, inj, retro=True)

        axes[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0) # image of staned axons at the source
        axes[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0) # image of staned axons at the target

        set_axis_labels(col_map, axes)
        fig.canvas.draw()
        return axes,
        
def circle(phi, radius_circ=0.3, offset=0.5):
    return np.array([radius_circ*np.cos(phi)+offset, radius_circ*np.sin(phi)+offset])

def set_axis_labels(col_map, axes):
    axes[0].set_title(col_map.source_name, size=15)
    axes[0].set_xlabel(col_map.source_x, size=12)
    axes[0].set_ylabel(col_map.source_y, size=12)

    axes[1].set_title(col_map.target_name, size=15)
    axes[1].set_xlabel(col_map.target_y, size=12)
    axes[1].set_ylabel(col_map.target_x, size=12)  
    rc.fractional_axes(axes, Num, 'k')


for x in list(mutant_frames.keys()):
    try:
        mutations, retina, colliculus, rc, cc, = mutant_frames[x]
    except:
        print(f'{type(x) =}')
        print('Not a valid item.')
        continue

    for col_map in [cc]:
        lap_time = datetime.now()
        fig, axes = plt.subplots(ncols=2, figsize=(8,4))
        axes = axes.flat
        default_inject = [0.5,0.5]

        src, trg = tri_injection(col_map.df, default_inject) # image of a simulated injection as seen in the source and the target

        axes[0].imshow(src, cmap='Greys_r', origin='lower', vmax=1, vmin=0)
        axes[1].imshow(trg, cmap='Greys_r', origin='lower', vmax=1, vmin=0)

        set_axis_labels(col_map, axes)

        figure_title = f'{mutations} - {col_map.name} - Retrograde'
        fig.suptitle(figure_title, size=20)
                
        ani = FuncAnimation(fig, update_figure, blit=False, repeat=False,
                        frames=np.linspace(0,2*np.pi,360, endpoint=None))

        f=f"{figure_title.replace('/','').replace(' ', '')}.mp4"
        print(f)
        writervideo = animation.FFMpegWriter(fps=30) 
        if not show_grads_bool: ani.save(f, writer=writervideo, dpi=300)
        print('\tLap Time: {}'.format(datetime.now() - lap_time))

        plt.close(fig)


print('Total Time: {}'.format(datetime.now() - start_time))


#%%