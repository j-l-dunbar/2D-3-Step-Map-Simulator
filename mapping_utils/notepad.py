# #%%
# import numpy as np
# import matplotlib.pyplot as plt


# num_rows = 20
# x = np.linspace(0,1,num_rows)
# EphB = x**2
# EphA = x**2
# coords = np.meshgrid(EphA, EphB)

# plt.scatter(*coords, s=5)

# plt.imshow(coords[0])
# # plt.imshow(coords[1])
# coords = np.array(coords).T


# print(coords.shape)
# print(coords[10,11])
# %%


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm      
# from matplotlib.widgets import Slider

# vals = np.linspace(-np.pi,np.pi,100)
# xgrid, ygrid = np.meshgrid(vals,vals)       

# def f(x, y, b):
#     return np.sin(x * b)
# b = 5

# ax = plt.subplot(111)
# plt.subplots_adjust(left=0.15, bottom=0.25)


# fig = plt.imshow(f(xgrid, ygrid, b), cm.gray)


# plt.axis('off')
# fig.axes.get_xaxis().set_visible(False)
# fig.axes.get_yaxis().set_visible(False)

# axb = plt.axes([0.15, 0.1, 0.65, 0.03])
# sb = Slider(axb, 'b', 0.001, 0.999, valinit=0.5)
# def update(val):
#     fig.set_data(f(xgrid, ygrid, val))
# sb.on_changed(update)

# plt.show()
# # %%

import matplotlib.pyplot as plt

def onclick(event):
    if event.button == 1:
         inj = [event.xdata, event.ydata]
    #clear frame
    plt.clf()
    plt.scatter(*inj); #inform matplotlib of the new data
    plt.draw() #redraw

fig,ax=plt.subplots()
ax.scatter(x,y)
fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()
plt.draw()
