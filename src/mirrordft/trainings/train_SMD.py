import wandb 
from transformers import HfArgumentParser
import os 
import sys 
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import shutil
import zipfile
from mirrordft.utils import *
from mirrordft.models import *

@dataclass
class TrainingConfig(UtilConfig):
    job_name: Optional[str] = field(default='MD')
    scf_compare: Optional[bool] = field(default=False)
    mu: Optional[float] = field(default=None)
    ratio: Optional[float] = field(default=0.5)
    device: Optional[str] = field(default='0')
    plot: Optional[bool] = field(default=False)
    decay: Optional[str] = field(default='exp')
    decay_iter: Optional[int] = field(default=1000)
    output_dir: Optional[str] = field(default=None)
parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)
# os.environ['CUDA_VISIBLE_DEVICES'] = device 

job_name = job_name + f"_D{dim}"

timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
problem_folder = f'D{dim}_N{int(N)}_L{int(L)}_b{int(beta*10)}_a{int(alpha*10)}_mu{int(mu*100)}'

tags = ['SMD', problem_folder, f'{dim}D']
model_name = f'SMD_c{int(cheat)}_Ns{N_samples}_Np{N_poles}'

dim_folder = f'{dim}D-cases'
if output_dir is not None:
    directory = os.path.join(output_dir,dim_folder, problem_folder, model_name, timestamp)
else:
    directory = os.path.join('./outputs',dim_folder, problem_folder, model_name, timestamp)
logger = GenLogger(directory, config, raw=raw)

if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 


# run SCF 
Ls = tuple(L for _ in range(dim))
Ns = tuple(N for _ in range(dim))

N_vec = np.prod(Ns)
# External potential
print(alpha, beta)

# centers = np.array([-1,4]).reshape(-1,1)
# masses = np.array([1.5, 0.5])
# ham.update_external_yukawa_centers(centers, masses)

if scf_compare:
    ham = deterministicHamiltonian(Ns, Ls, beta=beta, alpha=alpha, fourier=True, dense=True)
    ham.update_external_yukawa(ratio=ratio)
    print(f'Running SCF')
    res_scf = run_scf(ham, N_electrons, max_iter=1000, tol=1e-6, mu=mu, true_P_half=True)
    density_scf = res_scf['density']
    objective_scf  = res_scf['energy_free']
    print("Optimal Objective: ", objective_scf)

    if mu is None:
        mu = res_scf['mu']

    gold_algo = GoldAlgo(ham, density_scf, mu, N_samples)

# run MD
eval_history = defaultdict(list)
density_history = np.zeros([max_iter,N_vec])

stochastic_ham = StochasticHamiltonian(Ns, Ls,beta, alpha=alpha, N_poles=N_poles)

energy_keys = ['energy_kinetic', 'energy_external', 'entropy','sum_rho','energy_free','energy_yukawa']
if scf_compare:
    stochastic_ham.potential_external = ham.potential_external
else:
    stochastic_ham.update_external_yukawa(ratio=ratio)
c_H = -1/2
v_H = stochastic_ham.potential_external.copy()
stochastic_ham.update_poles_shifts(c_H, v_H) 


print(f'Running MD with mu={mu}')
start_time = time.time()
for iter in range(max_iter):

    # Compute the objective and gradient of the objective
    contour_start = time.time()
    dic_step = stochastic_ham.objective(c_H, v_H, N_samples=N_samples, tol=tol, mu=mu)
    contour_elapsed = time.time() - contour_start
    
    # Update v_H    
    grad_v_H = dic_step['grad_vH']
    if decay == 'sqrt' and iter > decay_iter:
        effective_lr = lr/beta/ np.sqrt(iter - decay_iter+1)
    elif decay == 'exp':
        effective_lr = lr/beta* np.exp(-iter/decay_iter)
    else:
        effective_lr = lr/beta
    v_H = (1-effective_lr)*v_H + effective_lr*(grad_v_H - mu)

    # Update density history
    # print(dic_step["energy_kinetic"])
    density_history[iter] = dic_step['density']

    # Update eval history
    for key in energy_keys:
        eval_history[key].append(dic_step[key])
        eval_history[f'half_{key}'].append(np.mean(eval_history[key][iter//2:iter+1]))
    eval_history["iter"].append(iter)
    last_half_average_density_MD = np.mean(density_history[iter//2:iter+1], axis=0)
    fake_energy_yukawa = eval_history["half_energy_yukawa"][-1]
    true_energy_yukawa = stochastic_ham.energy_yukawa(last_half_average_density_MD)
    eval_history["half_energy_yukawa"][-1]= true_energy_yukawa
    eval_history["half_energy_free"][-1] += true_energy_yukawa - fake_energy_yukawa

    # Compute density error
    if scf_compare:
        density_gold = gold_algo.step()
        norm_scf = np.linalg.norm(density_scf, ord=1)
        density_error = np.linalg.norm(last_half_average_density_MD - density_scf, ord=1)/norm_scf
        eval_history["density_error"].append(density_error)
        density_error_gold = np.linalg.norm(density_gold - density_scf, ord=1)/norm_scf
        eval_history["density_error_gold"].append(density_error_gold)

    # Update poles shifts
    if iter>0 and iter % update_poles_iter == 0:
        stochastic_ham.update_poles_shifts(c_H, v_H) 

    # Log info
    if iter>0 and iter % eval_iter == 0:

        current_time = time.time()
        info = f"Iteration: {iter}, " 
        info += f"Time: {current_time - start_time:.4f}s, " 
        info += f"Time for contour: {contour_elapsed:.4f}s, "
        info += f"Objective: {eval_history['half_energy_free'][-1]:.4e}, "
        info += f"LR: {effective_lr:.4e}, "
        info += f"Sum of Density: {eval_history['half_sum_rho'][-1]:.4e}, "
        if scf_compare:
            info += f"Density Error: {eval_history['density_error'][-1]:.4e}, "
            info += f"Density Error Gold: {eval_history['density_error_gold'][-1]:.4e}, "
            info += f"Optimal Objective: {objective_scf:.4e}"
        logger.info(info)

        if not raw:
            wanb_info = {
                "iter": iter,
                "objective": float(eval_history['energy_free'][-1]),
                "time_for_contour": float(contour_elapsed),
                "average_objective": float(eval_history['half_energy_free'][-1]),
                "average_sum_rho": float(eval_history['half_sum_rho'][-1]),
                "time": float(current_time - start_time),
                "sum_rho": float(eval_history['sum_rho'][-1]),
                "lr": float(effective_lr),
            }
            if scf_compare:
                wanb_info["optimal_obj"] = float(objective_scf)
                wanb_info["density_error"] = float(eval_history['density_error'][-1])
                wanb_info["density_error_gold"] = float(eval_history['density_error_gold'][-1])
            wandb.log(wanb_info)


logging.shutdown()

if scf_compare:
# Print a comparison of energy components between SCF and MD
    from tabulate import tabulate

    # Extract SCF energy components
    scf_energy_free = res_scf.get('energy_free', 'N/A')
    scf_energy_kinetic = res_scf.get('energy_kinetic', 'N/A')
    scf_energy_yukawa = res_scf.get('energy_yukawa', 'N/A')
    scf_energy_external = res_scf.get('energy_external', 'N/A')
    scf_energy_entropy = res_scf.get('entropy', 'N/A')
    scf_sum_rho = np.sum(density_scf) if density_scf is not None else 'N/A'

    # Extract last MD step energy components
    md_energy_free = eval_history['half_energy_free'][-1]
    md_energy_kinetic = eval_history['half_energy_kinetic'][-1]
    md_energy_yukawa = eval_history['half_energy_yukawa'][-1]
    md_energy_external = eval_history['half_energy_external'][-1]
    md_energy_entropy = eval_history['half_entropy'][-1]
    md_sum_rho = eval_history['half_sum_rho'][-1]

    # Create comparison table
    table_data = [
        ["Energy Free", scf_energy_free, md_energy_free],
        ["Energy Kinetic", scf_energy_kinetic, md_energy_kinetic],
        ["Energy Yukawa", scf_energy_yukawa, md_energy_yukawa],
        ["Energy External", scf_energy_external, md_energy_external],
        ["Energy Entropy", scf_energy_entropy, md_energy_entropy],
        ["Sum of Density", scf_sum_rho, md_sum_rho]
    ]

    # Print the table
    print("\nComparison of SCF vs MD energy components:")
    print(tabulate(table_data, headers=["Component", "SCF", "MD"], tablefmt="grid"))
    print("\n")

    with open(os.path.join(directory, "obj.log"), "w") as f:
        f.write("Comparison of SCF vs MD energy components:\n")
        f.write(tabulate(table_data, headers=["Component", "SCF", "MD"], tablefmt="grid"))
        f.write("\n")
    print(f"Saved comparison table to {os.path.join(directory, 'obj.log')}")


if not raw:
    wandb.finish()
    data_path = os.path.join(directory, "data")
    os.makedirs(data_path, exist_ok=True)
    if scf_compare:
        data_dict = {
            "eval_history": eval_history,
            "res_scf": res_scf,
            "density_history": density_history,
            "external_potential": stochastic_ham.potential_external,
            "config": config_dict,
            "density_scf": density_scf,
            "density_gold": density_gold,
            "density_md": last_half_average_density_MD,
        }
    else:
        data_dict = {
            "eval_history": eval_history,
            "density_history": density_history,
            "external_potential": stochastic_ham.potential_external,
            "config": config_dict,
            "density_md": last_half_average_density_MD,
        }
    with open(os.path.join(data_path, "data.pkl"), "wb") as f:
        pickle.dump(data_dict, f)

if  plot==True: 

    # Generate plots using the plotting utility
    print("\nGenerating plots...")
    try:
        import subprocess
        if dim == 1:
            plot_command = f"python -m src.plots.plot_1D --stamp={timestamp}"
        elif dim == 2:
            plot_command = f"python -m src.plots.plot_2D --stamp={timestamp}"
        elif dim == 3:
            plot_command = f"python -m src.plots.plot_3D --stamp={timestamp}"
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
        print(f"Running: {plot_command}")
        result = subprocess.run(plot_command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("Plots generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during plot generation: {str(e)}")
