import sys
import os
import pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')
sys.path.append(src_path)

from mirrordft.models.hamiltonian import *
from mirrordft.plots.plotter import *

# @dataclass
# class PlotConfig:
#     stamp: Optional[str] = field(default='20250309-072304')
# load the stamp
parser = HfArgumentParser(PlotConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)

# locate the stamp folder
path_stamp = find_stamp_folder(stamp)
# 
if path_stamp is not None:

    # Load the data
    print("Loading data...")
    data_path = os.path.join(path_stamp, 'data')
    data_dict = pickle.load(open(os.path.join(data_path, 'data.pkl'), 'rb'))
    config_dict = json.load(open(os.path.join(path_stamp, 'config.json'), 'r'))
    locals().update(config_dict)

    res_scf = data_dict['res_scf']
    density_history = data_dict['density_history']
    eval_history = data_dict['eval_history']
    if 'ham' in data_dict:
        ham = data_dict['ham']
        external_potential = ham.potential_external
    else:
        external_potential = data_dict['external_potential']
    volume = L**dim
    density_scf = res_scf['density']
    density_md = data_dict['density_md']
    density_gold = data_dict['density_gold']

    directory = os.path.join(path_stamp, 'figures')
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())


    if scf_compare:
        print("Plotting...")
        print(f"stamp: {stamp}", f"Dim: {dim}", f"N: {N}", f"L: {L}", f"beta: {beta}", f"alpha: {alpha}", f"mu: {mu}", f"ratio: {ratio}", f"N_samples: {N_samples}", f"N_poles: {N_poles}", f"lr: {lr}", f"max_iter: {max_iter}")

        if title:
            fig_size = (11.5,7) 
        else: 
            fig_size = (10, 5.5)
        plt.figure(figsize=fig_size,dpi=300)

        gs = gridspec.GridSpec(2,3)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[0,2])
        ax4 = plt.subplot(gs[1,0])
        ax5 = plt.subplot(gs[1,1])
        ax6 = plt.subplot(gs[1,2])

        # Plot 1 
        # External potential
        xs = np.linspace(0,L,N+1)[:-1] 
        ax1.plot(xs, external_potential,color='blue')
        ax1.set_xlabel("x")
        ax1.set_title("External potential")
        ax1.grid(True)

        # Plot 2
        # Density   
        ax2.plot(xs, density_scf, label='SCF',color='red', linestyle='--')
        ax2.plot(xs, density_md, label='MD',color='blue', linestyle='--')
        # ax2.plot(xs, density_gold, label='Gold',color='orange', linestyle='--')
        ax2.set_xlabel("x")
        ax2.set_title("Density")
        ax2.grid(True)
        my_legend(ax2,loc=1)

        # Plot 3
        # Density error
        ax3.semilogy(eval_history["iter"], eval_history["density_error"], label='MD',color='blue')
        ax3.semilogy(eval_history["iter"], eval_history["density_error_gold"], label='Gold',color='orange')
        ax3.set_title("Relative density error")
        ax3.set_xlabel("Iteration")
        ax3.grid(True)
        my_legend(ax3)
        
        # Plot 4
        # Hartree energy density
        x_data = eval_history["iter"]
        y_data = np.array(eval_history["half_energy_yukawa"])/volume
        optimal_value = res_scf['energy_yukawa']/volume
        ax4.plot(x_data, y_data, label='MD',color='blue')
        ax4.hlines( optimal_value, 0, max(x_data), colors='r', linestyles='--', label='optimal')
        ax4.set_title("Hartree energy density")
        ax4.set_xlabel("Iteration")
        ax4.grid(True)  
        my_inset_plot(ax4, x_data, y_data, optimal_value)

        # Plot 5
        # Free energy density
        y_data = np.array(eval_history["half_energy_free"])/volume
        optimal_value = res_scf['energy_free']/volume
        ax5.plot(x_data, y_data, label='MD',color='blue')
        ax5.hlines(optimal_value, 0, max(x_data), colors='r', linestyles='--', label='Optimal')
        max_abs_dist = max(abs(y_data - optimal_value))
        ax5.set_ylim([optimal_value - max_abs_dist/np.sqrt(7), optimal_value + max_abs_dist*np.sqrt(7)])
        ax5.set_title("Free energy density")
        ax5.set_xlabel("Iteration")
        ax5.grid(True)
        my_inset_plot(ax5, x_data, y_data, optimal_value)

        # Plot 6
        # Sum of density
        y_data = np.array(eval_history["half_sum_rho"])/volume
        optimal_value = sum(density_scf)/volume
        ax6.plot(x_data, y_data, label='MD',color='blue')
        ax6.hlines(optimal_value, 0, max(x_data), colors='r', linestyles='--', label=r'Optimal')
        ax6.set_title("Electrons per unit volume")
        ax6.set_xlabel("Iteration")
        ax6.grid(True)
        my_inset_plot(ax6, x_data, y_data, optimal_value)
        if title:
            plt.suptitle(f"stamp={stamp}, \n Dim={dim}, N={N}, L={L}, beta={beta}, alpha={alpha}, mu={mu}, ratio={ratio}, \n N_samples={N_samples}, N_poles={N_poles}, lr={lr}, max_iter={max_iter}, decay={decay}, decay_iter={decay_iter}")
        plt.tight_layout()
        if save_path is not None:
            directory = save_path
        if file_name is not None:
            save_name = file_name
        else:
            save_name = f"1D_res_{timestamp}"
        plt.savefig(os.path.join(directory, f"{save_name}.png"))
        plt.savefig(os.path.join(directory, f"{save_name}.pdf"), format='pdf')
        # plt.savefig(os.path.join(directory, f"{save_name}.eps"), format='eps')
        plt.close()
        
        print(f"Saved to {os.path.join(directory, f'{save_name}.png')}")
        print(f"Saved to {os.path.join(directory, f'{save_name}.pdf')}")
        # print(f"Saved to {os.path.join(directory, f'{save_name}.eps')}")