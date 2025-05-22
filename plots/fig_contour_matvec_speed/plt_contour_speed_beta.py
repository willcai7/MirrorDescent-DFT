import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
with open('./figures/fig_contour_matvec_speed/data_beta.pkl', 'rb') as f:
    data = pickle.load(f)

betas = data['betas']
info = data['info']


plt.figure(figsize=(12, 10), dpi=300)
# grid_size = ['30003','101x101','11x11x11']
colors = ['firebrick','dodgerblue' ,'orange']
plt.subplot(2,2,1)
dim= 0
    # times = np.array(data[dim,:,0])
mean_times = info[dim,:,0]
std_times = info[dim,:,1]
plt.loglog(betas, mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
plt.fill_between(betas, 
                    mean_times - std_times,
                    mean_times + std_times,
                    color=colors[dim],
                    alpha=0.3)
plt.grid(True)
plt.xlabel(r'Inverse temperature $\beta$')
plt.ylabel('Time (s)')
plt.ylim(5e-4, 5e-2)
plt.title('Scaling of 1D')

plt.subplot(2,2,2)
dim= 1
    # times = np.array(data[dim,:,0])
mean_times = info[dim,:,0]
std_times = info[dim,:,1]
plt.loglog(betas, mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
plt.fill_between(betas, 
                mean_times - std_times,
                mean_times + std_times,
                color=colors[dim],
                alpha=0.3)
plt.grid(True)
plt.xlabel(r'Inverse temperature $\beta$')
plt.ylabel('Time (s)')
plt.ylim(5e-4, 5e-2)
plt.title('Scaling of 2D')

plt.subplot(2,2,3)
dim= 2
    # times = np.array(data[dim,:,0])
mean_times = info[dim,:,0]
std_times = info[dim,:,1]
plt.loglog(betas, mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
plt.fill_between(betas, 
                mean_times - std_times,
                mean_times + std_times,
                color=colors[dim],
                alpha=0.3)
plt.grid(True)
plt.xlabel(r'Inverse temperature $\beta$')
plt.ylabel('Time (s)')
plt.ylim(5e-4, 5e-2)
plt.title('Scaling of 3D')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.subplot(2,2,4)
for dim in range(3):
    # times = np.array(data[dim,:,0])
    mean_errors = info[dim,:,2]
    std_errors = info[dim,:,3]
    plt.semilogy(betas, mean_errors, label=f'{dim+1}D', color=colors[dim], linewidth=2)
    plt.fill_between(betas, 
                    mean_errors - std_errors,
                    mean_errors + std_errors,
                    color=colors[dim],
                    alpha=0.3)

plt.xlabel(r'Inverse temperature $\beta$')
plt.ylabel('Error')
plt.title('Error of the contour integration method')
plt.legend(frameon=True, edgecolor='black', loc='best', bbox_to_anchor=None, fancybox=False)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./figures/fig_contour_matvec_speed/contour_scaling_beta_{timestamp}.png')  

