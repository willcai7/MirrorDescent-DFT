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


# load the stamp
parser = HfArgumentParser(PlotConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)

# locate the stamp folder
path_stamp = find_stamp_folder(stamp)

if path_stamp is not None: 

    print("Loading data...")
    data_path = os.path.join(path_stamp, 'data')
    data_dict = pickle.load(open(os.path.join(data_path, 'data.pkl'), 'rb'))
    config_dict = json.load(open(os.path.join(path_stamp, 'config.json'), 'r'))
    locals().update(config_dict)

    density_history = data_dict['density_history']
    eval_history = data_dict['eval_history']
    volume = L**dim
    density_md = data_dict['density_md']
    if 'ham' in data_dict:
        ham = data_dict['ham']
        external_potential = ham.potential_external
    else:
        external_potential = data_dict['external_potential']
    directory = os.path.join(path_stamp, 'figures')
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())

    if scf_compare:

        res_scf = data_dict['res_scf']
        density_scf = res_scf['density']

        print("Plotting...")
        print(f"stamp: {stamp}", f"Dim: {dim}", f"N: {N}", f"L: {L}", f"beta: {beta}", f"alpha: {alpha}", f"mu: {mu}", f"ratio: {ratio}", f"N_samples: {N_samples}", f"N_poles: {N_poles}", f"lr: {lr}", f"max_iter: {max_iter}")
        plt.figure(figsize=(19, 7), dpi=300)
        # plt.rcParams['axes.linewidth'] = 2
        gs = plt.GridSpec(2, 4, figure=plt.gcf(), wspace=0.4, hspace=0.3, top=0.82)

        # External Potential
        ax1 = plt.subplot(gs[0, 0])
        data = external_potential.reshape(N, N, N)
        im1 = ax1.imshow(data[0,:,:], cmap='plasma')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("External potential")

        # SCF Density
        ax2 = plt.subplot(gs[0, 1])
        data = density_scf.reshape(N, N, N)
        im2 = ax2.imshow(data[0,:,:], cmap='viridis')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        vmax = 1.1 * max(np.max(density_scf), np.max(density_md))
        im2.set_clim(0, vmax)
        ax2.set_title("SCF density")

        # MD Density
        ax3 = plt.subplot(gs[0, 2])
        data = density_md.reshape(N, N, N)
        im3 = ax3.imshow(data[0,:,:], cmap='viridis')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        im3.set_clim(0, vmax)
        ax3.set_title("MD density")

        # Final Density Error
        ax4 = plt.subplot(gs[0, 3])
        data = density_md - density_scf
        data = data.reshape(N, N, N)
        im4 = ax4.imshow(data[0,:,:], cmap='grey')
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        # im4.set_clim(0, vmax)
        ax4.set_title("Final density error")

        # Density Error
        ax5 = plt.subplot(gs[1, 0])
        iters = np.linspace(1, max_iter, max_iter)
        ax5.semilogy(eval_history["iter"], eval_history["density_error"], label='MD', color='blue')
        ax5.semilogy(iters, eval_history["density_error_gold"], label='Gold', color='orange')
        ax5.set_title("Relative density error")
        ax5.set_xlabel("Iteration")
        ax5.grid(True)
        my_legend(ax5)

        # Objective
        x_data = np.linspace(1, max_iter, max_iter)
        y_data = np.array(eval_history["half_energy_yukawa"])/volume
        optimal_value = res_scf['energy_yukawa']/volume
        ax6 = plt.subplot(gs[1, 1])
        ax6.plot(x_data, y_data, label='MD', color='blue')
        ax6.hlines(optimal_value, 0, max(x_data), colors='r', linestyles='--', label='Optimal')
        ax6.set_title("Hartree energy density")
        # ax6.set_xlim([0, max(x_data)+10])
        ax6.set_xlabel("Iteration")
        ax6.grid(True)
        my_inset_plot(ax6, x_data, y_data, optimal_value)

        # Free Energy
        ax7 = plt.subplot(gs[1, 2])
        y_data = np.array(eval_history["half_energy_free"])/volume
        optimal_value = res_scf['energy_free']/volume
        ax7.plot(x_data, y_data, label='MD', color='blue')
        ax7.hlines(optimal_value, 0, max(x_data), colors='r', linestyles='--', label='Optimal')
        ax7.set_title("Free energy density")
        ax7.set_xlabel("Iteration")
        # ax7.set_xlim([0, max(x_data)+10])
        max_abs_dist = max(abs(y_data - optimal_value))
        ax7.set_ylim([optimal_value - max_abs_dist/np.sqrt(7), optimal_value + max_abs_dist*np.sqrt(7)])
        ax7.grid(True)
        my_inset_plot(ax7, x_data, y_data, optimal_value)

        # Sum of Density
        ax8 = plt.subplot(gs[1, 3])
        y_data = np.array(eval_history["half_sum_rho"])/volume
        optimal_value = sum(density_scf)/volume
        ax8.plot(x_data, y_data, label='MD', color='blue')
        ax8.hlines(optimal_value, 0, max(x_data), colors='r', linestyles='--', label=r'Optimal')
        ax8.set_title("Electrons per unit volume")
        ax8.set_xlabel("Iteration")
        # ax8.set_xlim([0, max(x_data)+10])
        ax8.grid(True)
        my_inset_plot(ax8, x_data, y_data, optimal_value)

        if title:
            plt.suptitle(f"stamp={stamp}, \n Dim={dim}, N={N}, L={L}, beta={beta}, alpha={alpha}, mu={mu}, ratio={ratio}, \n N_samples={N_samples}, N_poles={N_poles}, lr={lr}, max_iter={max_iter}, decay={decay}, decay_iter={decay_iter}")

        if save_path is not None:
            directory = save_path
        if file_name is not None:
            save_name = file_name
        else:
            save_name = f"3D_res_{timestamp}"
        plt.savefig(os.path.join(directory, f"{save_name}.png"))
        plt.savefig(os.path.join(directory, f"{save_name}.pdf"), format='pdf')
        plt.savefig(os.path.join(directory, f"{save_name}.eps"), format='eps')
        plt.close()
        
        print(f"Saved to {os.path.join(directory, f'{save_name}.png')}")
        print(f"Saved to {os.path.join(directory, f'{save_name}.pdf')}")
        print(f"Saved to {os.path.join(directory, f'{save_name}.eps')}")
    
    else:
        print("Plotting...")
        print(f"stamp: {stamp}", f"Dim: {dim}", f"N: {N}", f"L: {L}", f"beta: {beta}", f"alpha: {alpha}", f"mu: {mu}", f"ratio: {ratio}", f"N_samples: {N_samples}", f"N_poles: {N_poles}", f"lr: {lr}", f"max_iter: {max_iter}")
        plt.figure(figsize=(12, 7), dpi=300)
        # plt.rcParams['axes.linewidth'] = 2
        gs = plt.GridSpec(2, 3, figure=plt.gcf(), wspace=0.4, hspace=0.3, top=0.82)

        # External Potential
        ax1 = plt.subplot(gs[0, 1])
        data = external_potential.reshape(N, N, N)
        im1 = ax1.imshow(data[0,:,:], cmap='plasma')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("External potential")


        # MD Density
        ax3 = plt.subplot(gs[0, 2])
        data = density_md.reshape(N, N, N)
        im3 = ax3.imshow(data[0,:,:], cmap='viridis')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("MD density")

        # Objective
        x_data = np.linspace(1, max_iter, max_iter)
        y_data = np.array(eval_history["half_energy_yukawa"])/volume
        ax6 = plt.subplot(gs[1, 0])
        ax6.plot(x_data, y_data, label='MD', color='blue')
        ax6.set_title("Hartree energy density")
        ax6.set_xlabel("Iteration")
        ax6.grid(True)
        optimal_value = np.mean(y_data[int(len(y_data)*0.6):])
        my_inset_plot(ax6, x_data, y_data, optimal_value, show_optimal=False)

        # Free Energy
        ax7 = plt.subplot(gs[0, 0])
        y_data = np.array(eval_history["half_energy_free"])/volume
        ax7.plot(x_data, y_data, label='MD', color='blue')
        ax7.set_title("Free energy density")
        ax7.set_xlabel("Iteration")
        ax7.grid(True)
        optimal_value = np.mean(y_data[int(len(y_data)*0.6):])
        my_inset_plot(ax7, x_data, y_data, optimal_value, show_optimal=False)

        # Sum of Density
        ax8 = plt.subplot(gs[1, 1])
        y_data = np.array(eval_history["half_sum_rho"])/volume
        ax8.plot(x_data, y_data, label='MD', color='blue')
        ax8.set_title("Electrons per unit volume")
        ax8.set_xlabel("Iteration")
        ax8.grid(True)
        optimal_value = np.mean(y_data[int(len(y_data)*0.6):])
        my_inset_plot(ax8, x_data, y_data, optimal_value, show_optimal=False)

        # Entropy 
        ax9 = plt.subplot(gs[1, 2])
        y_data = np.array(eval_history["half_entropy"])/volume
        ax9.plot(x_data, y_data, label='MD', color='blue')
        ax9.set_title("Entropy per unit volume")
        ax9.set_xlabel("Iteration")
        ax9.grid(True)
        optimal_value = np.mean(y_data[int(len(y_data)*0.6):])
        my_inset_plot(ax9, x_data, y_data, optimal_value, show_optimal=False)

        if title:
            plt.suptitle(f"stamp={stamp}, \n Dim={dim}, N={N}, L={L}, beta={beta}, alpha={alpha}, mu={mu}, ratio={ratio}, \n N_samples={N_samples}, N_poles={N_poles}, lr={lr}, max_iter={max_iter}, decay={decay}, decay_iter={decay_iter}")

        plt.tight_layout()
        if save_path is not None:
            directory = save_path
        if file_name is not None:
            save_name = file_name
        else:
            save_name = f"3D_res_{timestamp}"
        plt.savefig(os.path.join(directory, f"{save_name}.png"))
        plt.savefig(os.path.join(directory, f"{save_name}.pdf"), format='pdf')
        plt.savefig(os.path.join(directory, f"{save_name}.eps"), format='eps')
        plt.close()
        
        print(f"Saved to {os.path.join(directory, f'{save_name}.png')}")
        print(f"Saved to {os.path.join(directory, f'{save_name}.pdf')}")
        print(f"Saved to {os.path.join(directory, f'{save_name}.eps')}")
