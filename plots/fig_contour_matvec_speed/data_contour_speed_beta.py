
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '6' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate

# scaling of N_samples
base_dim = 3
# Nss = [(base_dim**6,), 
#         (base_dim**3,base_dim**3), 
#         (base_dim**2, base_dim**2, base_dim**2)] # 1D, 2D, 3D
# Lss = [(base_dim**6-1,), 
#         ((base_dim**3-1)*np.sqrt(2), (base_dim**3-1)*np.sqrt(2)), 
#         ((base_dim**2-1)*np.sqrt(3), (base_dim**2-1)*np.sqrt(3), (base_dim**2-1)*np.sqrt(3))]  

Nss = [(101,), (31, 31), (11, 11, 11)]
Lss = [(100,), (30, 30), (10, 10, 10)]
# Nss = [(601,)]
# Lss = [(600,)]

test_size = 20
N_poles = 20
N_samples = 20
betas = np.linspace(0, 10, 51)[1:]
betas2 = np.linspace(10, 100, 51)[1:]
betas = np.concatenate([betas, betas2])

info = np.zeros((len(Nss), len(betas), 4))


for dim in range(len(Nss)):
# for dim in range(1):
    for i in tqdm(range(len(betas))):
        beta = betas[i]
        Ns = Nss[dim]
        Ls = Lss[dim]
        info[dim, i, :] = benchmark_contour(Ns, Ls,N_poles, N_samples, beta, test_size=test_size, gen_N_poles=True, verbose=False,tol=1e-5, check_error=True) 

for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of beta')
    headers = ["beta", "Mean Time (s)", "Std Time (s)", "Mean Error", "Std Error"]
    table = [[betas[i], info[dim, i, 0], info[dim, i, 1], info[dim, i, 2], info[dim, i, 3]] for i in range(len(betas))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))

data = {'info': info, 'betas': betas}
with open('./figures/fig_contour_matvec_speed/data_beta.pkl', 'wb') as f:
    pickle.dump(data, f)


