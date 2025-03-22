
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
import numpy as np
from benchmarks.benchmark_tool import *
import pickle
from tabulate import tabulate


Ns = (31,31)
Ls = (60,60)
N_poles = 20
N_samples = 20
beta = 10
test_size = 20
elapsed_time, errors = benchmark_contour(Ns, Ls,N_poles, N_samples, beta, test_size=test_size, check_error=True,gen_N_poles=True, verbose=True,tol=1e-5,ratio=1, alpha=0.5)
print(elapsed_time) 
print(errors)


