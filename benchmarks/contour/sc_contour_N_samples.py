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

# scaling of N_samples
Nss = [(101,),(11,11), (5, 5, 5)] # 1D, 2D, 3D
Lss = [(10,), (10,10), (10, 10, 10)] # 1D, 2D, 3D
test_size = 50  
beta = 10
N_poles = 100
N_sampless = np.linspace(0, 300, 31).astype(int)[1:]
elapsed_time_history = defaultdict(list)

for dim in range(len(Nss)):
    print(f'Scaling of {dim+1}D contour solver')
    Ns = Nss[dim]
    Ls = Lss[dim]
    for i in tqdm(range(len(N_sampless))):
        N_samples = N_sampless[i]
        elapsed_time = benchmark_contour(Ns, Ls,N_poles, N_samples, beta, test_size=test_size)
        elapsed_time_history[dim].append(elapsed_time)

for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of N_samples')
    times = np.array(elapsed_time_history[dim])
    headers = ["N_samples", "Mean Time (s)", "Std Time (s)"]
    table = [[N_sampless[i], times[i][0], times[i][1]] for i in range(len(N_sampless))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))

data = {'elapsed_time_history': elapsed_time_history, 'N_sampless': N_sampless}
with open('./history_N_samples.pkl', 'wb') as f:
    pickle.dump(data, f)

