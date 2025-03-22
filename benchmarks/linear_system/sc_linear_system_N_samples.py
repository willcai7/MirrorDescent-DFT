
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate

# Scaling of the inverse linear system solver

# scaling of N_samples
Nss = [(101,),(11,11), (5, 5, 5)] # 1D, 2D, 3D
Lss = [(10,), (10,10), (10, 10, 10)] # 1D, 2D, 3D
test_size = 50  
N_sampless = np.linspace(0, 300, 31).astype(int)[1:]
elapsed_time_history = defaultdict(list)

for dim in range(len(Nss)):
    print(f'Scaling of {dim+1}D linear system solver')
    Ns = Nss[dim]
    Ls = Lss[dim]
    for i in tqdm(range(len(N_sampless))):
        N_samples = N_sampless[i]
        elapsed_time, _ = benchmark_linear_system(Ns, Ls, N_samples,test_size=test_size)
        elapsed_time_history[dim].append(elapsed_time)

for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of N_samples')
    times = np.array(elapsed_time_history[dim])
    headers = ["N_samples", "Mean Time (s)", "Std Time (s)"]
    table = [[N_sampless[i], times[i][0], times[i][1]] for i in range(len(N_sampless))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))

with open('./elapsed_time_history_N_samples.pkl', 'wb') as f:
    pickle.dump(elapsed_time_history, f)


