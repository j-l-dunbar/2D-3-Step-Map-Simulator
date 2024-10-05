#%%
import numpy as np
import matplotlib.pyplot as plt


num_rows = 20
x = np.linspace(0,1,num_rows)
EphB = x**2
EphA = x**2
coords = np.meshgrid(EphA, EphB)

plt.scatter(*coords, s=5)

plt.imshow(coords[0])
# plt.imshow(coords[1])
coords = np.array(coords).T


print(coords.shape)
print(coords[10,11])
# %%