
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate

# scaling of N_samples
Nss = [(101,), (31, 31), (11, 11, 11)]
Lss = [100,30,10]
# Nss = [(601,)]
# Lss = [(600,)]

test_size = 20
N_poles = 20
N_samples = 20
beta = 10

factors = np.linspace(0.1, 10, 100)
info = np.zeros((len(Nss), len(factors), 4)) 
L_info = np.zeros((len(Nss), len(factors)))

for dim in range(len(Nss)):
# for dim in range(1):
    for i in tqdm(range(len(factors))):
        Ns = Nss[dim]
        L = Lss[dim]
        Ls = tuple([L*factors[i] for _ in range(len(Ns))])
        info[dim, i, :] = benchmark_contour(Ns, Ls,N_poles, N_samples, beta, test_size=test_size, gen_N_poles=True, verbose=False,tol=1e-5, check_error=True) 
        L_info[dim, i] = Ls[0]
        
for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of L')
    headers = ["L factor", "L value", "Mean Time (s)", "Std Time (s)", "Mean Error", "Std Error"]
    table = [[factors[i], L_info[dim, i], info[dim, i, 0], info[dim, i, 1], info[dim, i, 2], info[dim, i, 3]] for i in range(len(factors))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))
    
    # Save table to txt file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'./figures/fig_contour_matvec_speed/table_L_scaling_{dim+1}D.txt', 'w') as f:
        f.write(f'{dim+1}D scaling of L\n')
        f.write(f'Generated on: {timestamp}\n')
        f.write(tabulate(table, headers=headers, floatfmt=".6f"))
data = {'info': info, 'L_info': L_info}
with open('./figures/fig_contour_matvec_speed/data_L.pkl', 'wb') as f:
    pickle.dump(data, f)


