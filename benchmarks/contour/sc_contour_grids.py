import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate

# scaling of N_samples
Nss = [[(51,),(101,),(151,), (201,), (251,), (301,), (401,)],
        [(5,5),(7,7),(9,9),(11,11), (13,13),(15,15), (17,17), (19,19), (21,21)], 
        [(3,3,3),(3,3,5), (3,5,5),(5, 5, 5), (5, 5, 7), (5, 7, 7), (7, 7, 7), (7, 7, 9)]] 
Lss = [(10,), (10,10), (10, 10, 10)] # 1D, 2D, 3D
test_size = 50  
beta = 10
N_poles = 100
N_samples = 100
N_grids = [[],[],[]]
elapsed_time_history = defaultdict(list)

for dim in range(len(Nss)):
    print(f'Scaling of {dim+1}D contour solver')
    Ns = Nss[dim]
    Ls = Lss[dim]
    for i in tqdm(range(len(Ns))):
        N = Ns[i]
        elapsed_time = benchmark_contour(N, Ls,N_poles, N_samples, beta, test_size=test_size)
        elapsed_time_history[dim].append(elapsed_time)
        N_grids[dim].append(np.prod(N))

for dim in range(len(Nss)):
    print(f'{dim+1}D scaling of N_grids')
    times = np.array(elapsed_time_history[dim])
    headers = ["N_grids", "Mean Time (s)", "Std Time (s)"]
    table = [[N_grids[dim][i], times[i][0], times[i][1]] for i in range(len(N_grids[dim]))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))

data = {'elapsed_time_history': elapsed_time_history, 'N_grids': N_grids}
with open('./history_N_grids.pkl', 'wb') as f:
    pickle.dump(data, f)


