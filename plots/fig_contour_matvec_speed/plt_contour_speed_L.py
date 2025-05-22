import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
with open('./figures/fig_contour_matvec_speed/data_L.pkl', 'rb') as f:
    data = pickle.load(f)

L_info = data['L_info']
info = data['info']


plt.figure(figsize=(12, 10), dpi=300)
# grid_size = ['30003','101x101','11x11x11']
colors = ['firebrick','dodgerblue' ,'orange']
plt.subplot(2,2,1)
dim= 0
    # times = np.array(data[dim,:,0])
mean_times = info[dim,:,0]
std_times = info[dim,:,1]
plt.plot(L_info[dim,:], mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
plt.fill_between(L_info[dim,:], 
                    mean_times - std_times,
                    mean_times + std_times,
                    color=colors[dim],
                    alpha=0.3)
plt.grid(True)
plt.xlabel(r'L')
plt.ylabel('Time (s)')
plt.title('Scaling of 1D')

plt.subplot(2,2,2)
dim= 1
    # times = np.array(data[dim,:,0])
mean_times = info[dim,:,0]
std_times = info[dim,:,1]
plt.plot(L_info[dim,:], mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
plt.fill_between(L_info[dim,:], 
                mean_times - std_times,
                mean_times + std_times,
                color=colors[dim],
                alpha=0.3)
plt.grid(True)
plt.xlabel(r'L')
plt.ylabel('Time (s)')
plt.title('Scaling of 2D')

plt.subplot(2,2,3)
dim= 2
    # times = np.array(data[dim,:,0])
mean_times = info[dim,:,0]
std_times = info[dim,:,1]
plt.plot(L_info[dim,:], mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
plt.fill_between(L_info[dim,:], 
                mean_times - std_times,
                mean_times + std_times,
                color=colors[dim],
                alpha=0.3)
plt.grid(True)
plt.xlabel(r'L')
plt.ylabel('Time (s)')
plt.title('Scaling of 3D')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.subplot(2,2,4)
for dim in range(3):
    # times = np.array(data[dim,:,0])
    mean_errors = info[dim,:,2]
    std_errors = info[dim,:,3]
    plt.semilogy(L_info[dim,:], mean_errors, label=f'{dim+1}D', color=colors[dim], linewidth=2)
    plt.fill_between(L_info[dim,:], 
                    mean_errors - std_errors,
                    mean_errors + std_errors,
                    color=colors[dim],
                    alpha=0.3)

plt.xlabel(r'L')
plt.ylabel('Error')
plt.title('Error of the contour integration method')
plt.legend(frameon=True, edgecolor='black', loc='best', bbox_to_anchor=None, fancybox=False)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./figures/fig_contour_matvec_speed/contour_scaling_L_{timestamp}.png')  

