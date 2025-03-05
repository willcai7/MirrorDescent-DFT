import wandb 
from transformers import HfArgumentParser
import os 
import sys 
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')
sys.path.append(src_path)

# print(sys.path)
from src.utils import *
from src.models import *

class TrainingConfig(UtilConfig):
    job_name: Optional[str] = field(default='MD')

parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)

timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
problem_folder = f'D{dim}L{int(L)}N{int(N)}b{int(beta*10)}a{int(alpha*10)}Ne{N_electrons}'

if cheat:
    tags = ['CMD', problem_folder]
    model_name = f'CMD_c{int(cheat)}Ns{N_samples}Np{N_poles}'
else:
    tags = ['MD', problem_folder]
    model_name = f'MD_c{int(cheat)}Ns{N_samples}Np{N_poles}'

directory = os.path.join('./outputs',problem_folder, model_name, timestamp)
logger = GenLogger(directory, config, raw=raw)

if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 


# run SCF 
Ls = tuple(L for _ in range(dim))
Ns = tuple(N for _ in range(dim))

N_vec = np.prod(Ns)
# External potential
centers = np.array([-1,4]).reshape(-1,1)
masses = np.array([1.5, 0.5])
print(alpha, beta)
ham = deterministicHamiltonian(Ns, Ls, beta=beta, alpha=alpha, fourier=True, dense=True)
ham.update_external_yukawa(centers, masses)


res_scf = run_scf(ham, N_electrons, max_iter=1000, tol=1e-6)
density_scf = res_scf['density']
objective_scf = res_scf['objective']
mu = res_scf['mu']

gold_algo = GoldAlgo(ham, density_scf, mu, N_samples)
# run MD
H = np.zeros([N_vec, N_vec])
eye = np.eye(N_vec)
iter = 1
eval_history = defaultdict(list)
density_history = np.zeros([max_iter,N_vec])

start_time = time.time()
for iter in range(max_iter):
    
    gradient,density_MD = ham.gradient(H, cheat=cheat, N_samples=N_samples)
    density_gold = gold_algo.step()
    effective_lr = lr/beta
    H = (1-effective_lr)*H + effective_lr*(gradient - mu*eye)
    density_history[iter] = density_MD
    last_half_average_density_MD = np.mean(density_history[iter//2:iter], axis=0)
    density_error = np.linalg.norm(last_half_average_density_MD - density_scf)
    eval_history["density_error"].append(density_error)
    density_error_gold = np.linalg.norm(density_gold - density_scf)
    eval_history["density_error_gold"].append(density_error_gold)
    if iter>0 and iter % eval_iter == 0:
        current_time = time.time()
        eval_history["iter"].append(iter)
        objective = ham.objective(H=H,mu=mu)
        for key, value in objective.items():
            eval_history[key].append(value)

        logger.info(
            f"Iteration: {iter}, "
            f"density error: {density_error}, "
            f"objective: {objective['objective']}, "
            f"Time: {current_time - start_time}"
        )

        if not raw:
            wandb.log({
                "iter": iter,
                "density_error": float(density_error),
                "density_error_gold": float(density_error_gold),
                "objective": float(objective['objective']),
                "time": float(current_time - start_time),
                "sum_rho": float(sum(density_MD)),
                "optimal_obj": float(objective_scf),
            })


logging.shutdown()

if not raw:
    wandb.finish()
    with open(os.path.join(directory, "eval_history.pkl"), "wb") as f:
        pickle.dump(eval_history, f)
    with open(os.path.join(directory, "res_scf.pkl"), "wb") as f:
        pickle.dump(res_scf, f)

    density_history_path = os.path.join(directory, "density_history.npy")
    np.save(density_history_path, density_history, allow_pickle=True)

if dim == 1:    
    plt.figure(figsize=(10,8))
    xs = np.linspace(0,L,N+1)[:-1]
    plt.subplot(2,2,1)
    plt.plot(xs, density_scf, label='SCF')
    plt.plot(xs, last_half_average_density_MD, label='MD')
    plt.plot(xs, density_gold, label='gold')
    plt.legend()
    plt.title("Density")
    plt.subplot(2,2,2)
    iters = np.linspace(1, max_iter, max_iter)
    plt.semilogy(iters, eval_history["density_error"], label='MD')
    plt.plot(iters, eval_history["density_error_gold"], label='gold')
    plt.title("Density Error")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(eval_history["iter"], eval_history["objective"], label='MD')
    plt.hlines(objective_scf, 0, max(eval_history["iter"]), colors='r', linestyles='--', label='optimal')
    plt.title("Objective")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(eval_history["iter"], eval_history["sum_rho"], label='MD')
    plt.hlines(N_electrons, 0, max(eval_history["iter"]), colors='r', linestyles='--', label=r'$N_{electron}$')
    plt.title("Sum of Density")
    plt.legend()
    plt.savefig(os.path.join(directory, "1D_res.png"))
    plt.close()





