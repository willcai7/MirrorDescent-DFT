import re
import os
import json
def find_stamp_folder(stamp, base_dir='./outputs'):
    for root, dirs, files in os.walk(base_dir):
        if stamp in os.path.basename(root):
            return os.path.relpath(root, start=os.getcwd())
    return None

def analyze_log_file(log_file_path):
    total_contour_time = 0
    count = 0
    total_time = None
    
    # Regular expressions
    contour_pattern = re.compile(r'Time for contour: (\d+\.\d+)s')
    total_time_pattern = re.compile(r'Time: (\d+\.\d+)s')
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Extract time for contour
            contour_match = contour_pattern.search(line)
            if contour_match:
                contour_time = float(contour_match.group(1))
                total_contour_time += contour_time
                count += 1
            
            # Extract total time (will keep updating until last line)
            time_match = total_time_pattern.search(line)
            if time_match:
                total_time = float(time_match.group(1))
    
    avg_contour_time = total_contour_time / count if count > 0 else 0
    
    return avg_contour_time, total_time

# Example usage
stamps = [    
    "20250309-220547",
    "20250309-220625",
    "20250309-220725",
    "20250311-044757",
    "20250310-173713",
    "20250310-173914",
    "20250310-004521",
    "20250310-004641",
    "20250310-004701",
    "20250311-044857",
    "20250310-175802",
    "20250310-180043",
    "20250310-183417",
    "20250310-183436",
    "20250310-183456",
    "20250310-183516",
    "20250320-153658",
    "20250320-155132",
    "20250320-213107",
    "20250320-215653",
    "20250320-215735",
    "20250312-125615",
    "20250312-125645",
    ]
for stamp in stamps:
    path_stamp = find_stamp_folder(stamp)
    log_file = f"{path_stamp}/training.log"
    json_file = f"{path_stamp}/config.json"
    avg_contour_time, total_time = analyze_log_file(log_file)
    with open(json_file, 'r') as f:
        config = json.load(f)
    N = config['N']
    L = config['L']
    dim = config['dim']
    beta = config['beta']
    alpha = config['alpha']
    N_vec = N**dim
    print("-------------------------------- ")
    print(f"Stamp: {stamp}")
    print(f"Dim: {dim}, N: {N}, L: {L}, beta: {beta}, alpha: {alpha}")
    print(f"Average time for contour: {avg_contour_time:.4f} seconds")
    print(f"Average time for contour per grid point: {avg_contour_time/N_vec:.4e} seconds")
    print("--------------------------------")