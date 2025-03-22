import os 
import subprocess

import glob
from pathlib import Path

# Function to find all stamp folders with a data subfolder
def collect_stamp_folders(dim,figure_exist=False):
    # Get the base directory for 1D cases
    base_dir = f"outputs/{dim}D-cases"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return []
    
    # Find all potential stamp folders
    stamp_folders = []
    
    # Walk through all subdirectories in the base directory
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            # Construct the full path to the potential stamp folder
            stamp_folder = os.path.join(root, dir_name)
            
            # Check if this folder has a data subfolder
            data_folder = os.path.join(stamp_folder, "data")
            figure_folder = os.path.join(stamp_folder, "figures")
            if os.path.exists(data_folder) and os.path.isdir(data_folder):
                if figure_exist:
                    stamp_folders.append(dir_name)
                else:
                    if not os.path.exists(figure_folder):
                        stamp_folders.append(dir_name)
    
    return stamp_folders

# Collect all valid stamp folders
dims = [1,2,3]
for dim in dims:
    timestamps = collect_stamp_folders(dim,figure_exist=True)
    if not timestamps:
        print("No valid stamp folders found with data subfolders")
    print(timestamps)

    for timestamp in timestamps:
        plot_command = f"python -m src.plots.plot_{dim}D --stamp={timestamp}"
        print(f"Running plot command for {timestamp}...")
        try:
            subprocess.run(plot_command, shell=True, check=True)
            print(f"Successfully plotted data for {timestamp}")
        except subprocess.CalledProcessError as e:
            print(f"Error plotting data for {timestamp}: {e}")