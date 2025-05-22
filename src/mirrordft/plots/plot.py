from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable
import matplotlib.pyplot as plt
import numpy as np
from transformers import HfArgumentParser
import os
import pickle
import json
import sys  
import json
import time

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')
sys.path.append(src_path)

from mirrordft.models.hamiltonian import *

@dataclass
class PlotConfig:
    stamp: Optional[str] = field(default='20250307-153145')
    

parser = HfArgumentParser(PlotConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)

def find_stamp_folder(stamp, base_dir='./outputs'):
    for root, dirs, files in os.walk(base_dir):
        if stamp in os.path.basename(root):
            return os.path.relpath(root, start=os.getcwd())
    return None

def my_legend():
    legend = plt.legend()
    # Set the legend frame linewidth to 2
    legend.get_frame().set_linewidth(legend_linewidth)
    legend.get_frame().set_edgecolor('black')

path_stamp = find_stamp_folder(stamp) 
if path_stamp is not None:

    # load data
    print("Loading data...")
    data_path = os.path.join(path_stamp, 'data')
    data_dict = pickle.load(open(os.path.join(data_path, 'data.pkl'), 'rb'))
    config_dict = json.load(open(os.path.join(path_stamp, 'config.json'), 'r'))
    res_scf = data_dict['res_scf']
    density_history = data_dict['density_history']
    eval_history = data_dict['eval_history']
    stochastic_ham = data_dict['stochastic_ham']
    ham = data_dict['ham']
    volume = np.prod(ham.Ls)
    locals().update(config_dict)
    density_scf = res_scf['density']
    last_half_average_density_MD = np.mean(density_history[-int(max_iter/2):], axis=0)
    # plot 
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    legend_linewidth = 1.5

    directory = os.path.join(path_stamp, 'figures')
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if dim == 1 and scf_compare:
        print("Plotting...")
        print(f"stamp: {stamp}", f"Dim: {dim}", f"N: {N}", f"L: {L}", f"beta: {beta}", f"alpha: {alpha}", f"mu: {mu}", f"ratio: {ratio}", f"N_samples: {N_samples}", f"N_poles: {N_poles}", f"lr: {lr}", f"max_iter: {max_iter}")
        plt.figure(figsize=(10,6),dpi=300)
        # Set all subplot borders to have linewidth 2
        xs = np.linspace(0,L,N+1)[:-1]

        plt.subplot(2,3,2)
        plt.plot(xs, density_scf, label='SCF',color='red', linestyle='--')
        plt.plot(xs, last_half_average_density_MD, label='MD',color='blue', linestyle='--')
        plt.xlabel("x")
        my_legend()
        plt.grid(True)
        plt.title("Density")

        plt.subplot(2,3,1)
        plt.plot(xs, ham.potential_external,color='blue')
        plt.xlabel("x")
        plt.grid(True)
        plt.title("External potential")

        plt.subplot(2,3,3)
        iters = np.linspace(1, max_iter, max_iter)
        plt.semilogy(eval_history["iter"], eval_history["density_error"], label='MD',color='blue')
        plt.semilogy(iters, eval_history["density_error_gold"], label='Gold',color='orange')
        plt.title("Relative density error")
        plt.xlabel("Iteration")
        plt.grid(True)
        my_legend()
        
        plt.subplot(2,3,4)
        plt.plot(eval_history["iter"], eval_history["half_energy_yukawa"]/volume, label='MD',color='blue')
        plt.hlines( res_scf['energy_yukawa']/volume, 0, max(eval_history["iter"]), colors='r', linestyles='--', label='optimal')
        plt.title("Hartree energy density")
        plt.xlabel("Iteration")
        plt.grid(True)
        
        plt.subplot(2,3,6)
        plt.plot(eval_history["iter"], eval_history["half_sum_rho"]/volume, label='MD',color='blue')
        plt.hlines(sum(density_scf)/volume, 0, max(eval_history["iter"]), colors='r', linestyles='--', label=r'Optimal')
        plt.title("Free energy density")
        plt.xlabel("Iteration")
        plt.grid(True)
        my_legend()

        plt.subplot(2,3,5)
        plt.plot(eval_history["iter"], eval_history["half_energy_free"]/volume, label='MD',color='blue')
        plt.hlines(res_scf['energy_free']/volume, 0, max(eval_history["iter"]), colors='r', linestyles='--', label='Optimal')

        plt.title("Electrons per unit volume")
        plt.xlabel("Iteration")
    
        plt.grid(True)

        plt.suptitle(f"stamp={stamp}, \n Dim={dim}, N={N}, L={L}, beta={beta}, alpha={alpha}, mu={mu}, ratio={ratio}, \n N_samples={N_samples}, N_poles={N_poles}, lr={lr}, max_iter={max_iter}, decay={decay}, decay_iter={decay_iter}")
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"1D_res_{timestamp}.png"))
        plt.close()
        print(f"Saved to {os.path.join(directory, f'1D_res_{timestamp}.png')}")

    elif dim == 2 and scf_compare:
        print("Plotting...")
        print(f"stamp: {stamp}", f"Dim: {dim}", f"N: {N}", f"L: {L}", f"beta: {beta}", f"alpha: {alpha}", f"mu: {mu}", f"ratio: {ratio}", f"N_samples: {N_samples}", f"N_poles: {N_poles}", f"lr: {lr}", f"max_iter: {max_iter}")
        plt.figure(figsize=(16, 6), dpi=300)
        plt.rcParams['axes.linewidth'] = 2
        gs = plt.GridSpec(2, 4, figure=plt.gcf(), wspace=0.4, hspace=0.3)

        # External Potential
        ax1 = plt.subplot(gs[0, 0])
        im1 = ax1.imshow(ham.potential_external.reshape(N, N), cmap='plasma')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("External potential")

        # SCF Density
        ax2 = plt.subplot(gs[0, 1])
        im2 = ax2.imshow(density_scf.reshape(N, N), cmap='viridis')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        vmax = 1.1 * max(np.max(density_scf), np.max(last_half_average_density_MD))
        im2.set_clim(0, vmax)
        ax2.set_title("SCF density")

        # MD Density
        ax3 = plt.subplot(gs[0, 2])
        im3 = ax3.imshow(last_half_average_density_MD.reshape(N, N), cmap='viridis')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        im3.set_clim(0, vmax)
        ax3.set_title("MD density")

        # Final Density Error
        ax4 = plt.subplot(gs[0, 3])
        diff = last_half_average_density_MD - density_scf
        im4 = ax4.imshow(diff.reshape(N, N), cmap='grey')
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
        my_legend()

        # Objective
        ax6 = plt.subplot(gs[1, 1])
        ax6.plot(eval_history["iter"], eval_history["half_energy_yukawa"]/volume, label='MD', color='blue')
        ax6.axhline(y=res_scf['energy_yukawa']/volume, color='r', linestyle='--', label='Optimal')
        ax6.set_title("Hartree energy (per unit)")
        ax6.set_xlabel("Iteration")
        ax6.grid(True)
        my_legend()

        # Free Energy
        ax7 = plt.subplot(gs[1, 2])
        ax7.plot(eval_history["iter"], eval_history["half_energy_free"]/volume, label='MD', color='blue')
        ax7.axhline(y=res_scf['energy_free']/volume, color='r', linestyle='--', label='Optimal')
        ax7.set_title("Free energy (per unit)")
        ax7.set_xlabel("Iteration")
        ax7.grid(True)
        my_legend()

        # Sum of Density
        ax8 = plt.subplot(gs[1, 3])
        ax8.plot(eval_history["iter"], eval_history["half_sum_rho"]/volume, label='MD', color='blue')   
        ax8.axhline(y=sum(density_scf)/volume, color='r', linestyle='--', label=r'Optimal')
        ax8.set_title("Sum of density (per unit)")
        ax8.set_xlabel("Iteration")
        ax8.grid(True)
        my_legend()

        plt.suptitle(f"Dim={dim}, N={N}, L={L}, beta={beta}, alpha={alpha}, N_samples={N_samples}, N_poles={N_poles}, mu={mu}, ratio={ratio}, lr={lr}, max_iter={max_iter}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle with rect parameter

        plt.savefig(os.path.join(directory, f"{dim}D_res_{timestamp}.png"))
        plt.close()
        print(f"Saved to {os.path.join(directory, f'{dim}D_res_{timestamp}.png')}")



