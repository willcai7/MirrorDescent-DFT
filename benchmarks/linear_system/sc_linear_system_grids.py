
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
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
Nss = [[(51,),(101,),(151,), (201,), (251,), (301,), (401,)],
        [(5,5),(7,7),(9,9),(11,11), (13,13),(15,15), (17,17), (19,19), (21,21)], 
        [(3,3,3),(3,3,5), (3, 5, 5),(5, 5, 5), (5, 5, 7), (5, 7, 7), (7, 7, 7), (7, 7, 9)]] 
        # 1D, 2D, 3D
Lss = [(10,), (10,10), (10, 10, 10)] # 1D, 2D, 3D
test_size = 50  
N_samples = 10
elapsed_time_history = defaultdict(list)
N_grids = [[],[],[]]

for dim in range(len(Nss)):
    print(f'Scaling of {dim+1}D linear system solver')
    Ns = Nss[dim]
    Ls = Lss[dim]
    for i in tqdm(range(len(Ns))):
        N = Ns[i]
        elapsed_time, _ = benchmark_linear_system(N, Ls, N_samples,test_size=test_size)
        elapsed_time_history[dim].append(elapsed_time)
        N_grids[dim].append(np.prod(N))
for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of N_grids')
    times = np.array(elapsed_time_history[dim])
    headers = ["N_grids", "Mean Time (s)", "Std Time (s)"]
    table = [[N_grids[dim][i], times[i][0], times[i][1]] for i in range(len(N_grids[dim]))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))

data = {'elapsed_time_history': elapsed_time_history, 'N_grids': N_grids}
with open('./elapsed_time_history_N_grids.pkl', 'wb') as f:
    pickle.dump(data, f)


