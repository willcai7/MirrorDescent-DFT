import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


from src.plots.plotter import *


with open('./figures/fig_contour_matvec_speed/data_beta.pkl', 'rb') as f:
    data = pickle.load(f)

betas = data['betas']
info = data['info']


plt.figure(figsize=(10, 4), dpi=300)
colors = ['firebrick','dodgerblue' ,'orange']
plt.subplot(1,2,1)
for dim in range(3):
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
plt.title(r'(a) Time vs $\beta$')

with open('./figures/fig_contour_matvec_speed/data_L.pkl', 'rb') as f:
    data = pickle.load(f)

factors = np.linspace(0.1, 10, 100)
info = data['info']
plt.subplot(1,2,2)
for dim in range(3):
    # times = np.array(data[dim,:,0])
    mean_times = info[dim,:,0]
    std_times = info[dim,:,1]
    plt.plot(factors, mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
    plt.fill_between(factors, 
                    mean_times - std_times,
                    mean_times + std_times,
                    color=colors[dim],
                    alpha=0.3)
plt.grid(True)


plt.xlabel(r'L factor')
# plt.ylabel('Time (s)')
plt.title(r'(b) Time vs $L$')
plt.legend(frameon=True, edgecolor='black', loc='best', bbox_to_anchor=None, fancybox=False)
plt.grid(True)
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'./figures/fig_contour_matvec_speed/contour_scaling_{timestamp}.png')  
plt.savefig(f'./figures/fig_contour_matvec_speed/contour_scaling_{timestamp}.pdf', format='pdf')

