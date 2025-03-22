import pickle
import matplotlib.pyplot as plt
import numpy as np
with open('./history_N_poles.pkl', 'rb') as f:
    data = pickle.load(f)

N_poless = data['N_poless']
elapsed_time_history = data['elapsed_time_history']

plt.figure(figsize=(5, 4), dpi=300)
grid_size = ['101','11x11','5x5x5']
colors = ['firebrick','dodgerblue' ,'orange']
for dim in range(len(elapsed_time_history)):
    times = np.array(elapsed_time_history[dim])
    mean_times = times[:,0]
    std_times = times[:,1]
    plt.plot(N_poless, mean_times, label=f'{dim+1}D', color=colors[dim], linewidth=2)
    plt.fill_between(N_poless, 
                    mean_times - std_times,
                    mean_times + std_times,
                    color=colors[dim],
                    alpha=0.3)

plt.xlabel('Number of poles')
plt.ylabel('Time (s)')
plt.title('Scaling of the contour integration method')
plt.legend(frameon=True, edgecolor='black', loc='best', bbox_to_anchor=None, fancybox=False)
plt.grid(True)
plt.tight_layout()
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2) 
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 15,    
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 15
})


plt.savefig('./contour_scaling_N_poles.png')

