#!/bin/bash
# Function to send email notification
send_email() {
    local to="willcai754@gmail.com"
    local subject="$1"
    local body="$2"
    local from="willcai754@gmail.com"  # Set your email address here
    
    {
        echo "From: $from"
        echo "To: $to"
        echo "Subject: $subject"
        echo ""
        echo "$body"
    } | msmtp --file=/home/jovyan/local/msmtp.conf -t "$to"
}



save_script_and_src() {
    # Get timestamp once and use it consistently
    local exp_uid=$1
    
    # Define directories with clear naming
    local output_dir="./outputs"
    local backup_dir="${output_dir}/backup"
    local backup_file="${backup_dir}/code_backup_${exp_uid}.tar.gz"
    
    # echo "Output directory: $output_dir"
    # echo "Backup directory: $backup_dir"
    # echo "Backup file: $backup_file"
    
    # Create directories with error checking
    # echo "Creating directories..."
    mkdir -p "$output_dir" || { echo "ERROR: Failed to create output directory"; return 1; }
    mkdir -p "$backup_dir" || { echo "ERROR: Failed to create backup directory"; return 1; }
    
    # Verify directories exist
    # echo "Verifying directories..."
    [ -d "$output_dir" ] || { echo "ERROR: Output directory doesn't exist after creation"; return 1; }
    [ -d "$backup_dir" ] || { echo "ERROR: Backup directory doesn't exist after creation"; return 1; }
    
    # echo "Creating tar archive..."
    # Use current directory as base for relative paths
    tar -czf "$backup_file" ./src/ ./exp_MD.sh || { echo "ERROR: tar command failed"; return 1; }

    # Copy the exp_MD.sh script to the backup directory
    # echo "Copying exp_MD.sh to backup directory..."
    
    # Verify tar file was created
    if [ -f "$backup_file" ]; then
        echo "SUCCESS: Backup created at: $backup_file"
        # Use consistent path for final message
        # echo "Script and src folders compressed and saved to $backup_file"
    else
        echo "ERROR: Backup file not created: $backup_file"
        return 1
    fi
    
    return 0
}

find_gpu() {
    local visible_devices="${1:-0,1,2,3}"
    gpu_index=-1
    while [ $gpu_index -eq -1 ]; do
        # Create a temporary file
        tmp_file=$(mktemp)
        nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits > "$tmp_file"
        
        while IFS=',' read -r index utilization; do
            # Use grep instead of bash pattern matching
            if echo ",$visible_devices," | grep -q ",$index," && [ "$(echo "$utilization == 0" | bc -l)" -eq 1 ]; then
                gpu_index=$index
                break
            fi
        done < "$tmp_file"
        
        rm "$tmp_file"
        
        if [ $gpu_index -eq -1 ]; then
            # echo "No unused GPU found in visible devices ($visible_devices). Waiting 30 seconds..."
            sleep 30
        fi
    done
    echo $gpu_index
}

# Function to run a single experiment
run_experiment() {
    local dim="$1"
    local N="$2"
    local L="$3"
    local beta="$4"
    local alpha="$5"
    local N_samples="$6"
    local N_poles="$7"
    local max_iter="$8"
    local raw="$9"
    local ratio="${10}"
    local eval_iter="${11}"
    local update_poles_iter="${12}"
    local lr="${13}"
    local scf_compare="${14}"
    local mu="${15}"
    local tol="${16}"
    local decay="${17}"
    local decay_iter="${18}"
    local plot="${19}"
    local cuda_visible_devices="${20}"
    local exp_uid="${21}"

    # Make sure outputs/temp directory exists
    mkdir -p "outputs/temp"
    
    # Generate unique ID for this run
    export uid="$(date +%y%m%d-%H%M%S)"
    
    # Create output directory name
    export output_dir="./outputs"
    mkdir -p "$output_dir"

    # Sleep for a random time between 1-5 seconds before finding GPU
    sleep $(( RANDOM % 5 + 10 ))
    
    gpu_index=$(find_gpu "$cuda_visible_devices")
    
    echo "$gpu_index"

    # Run the experiment and capture both stdout and stderr
    (
        export CUDA_VISIBLE_DEVICES=$gpu_index
        python -m mirrordft.trainings.train_SMD \
            --job_name='SMD' \
            --dim=$dim \
            --N=$N \
            --L=$L \
            --beta=$beta \
            --alpha=$alpha \
            --cheat=True \
            --N_samples=$N_samples \
            --N_poles=$N_poles \
            --max_iter=$max_iter \
            --raw=$raw \
            --ratio=$ratio \
            --eval_iter=$eval_iter \
            --update_poles_iter=$update_poles_iter \
            --lr=$lr \
            --scf_compare=$scf_compare \
            --mu=$mu \
            --tol=$tol \
            --decay=$decay \
            --decay_iter=$decay_iter \
            --plot=$plot \
            --output_dir=$output_dir
    ) > "outputs/temp/exp_${uid}_$$.log" 2>&1 &
    
    # Return the PID of the background process
    echo $!
}

# Define hyperparameter configurations
exp_uid="$(date +%y%m%d-%H%M%S)"
# cuda_visible_devices="0,1,2,3,4,5,6,7"
cuda_visible_devices="0,1,2,3"
save_script_and_src $exp_uid

# Format: "dim N L beta alpha N_samples N_poles max_iter raw ratio eval_iter update_poles_iter lr scf_compare mu tol decay decay_iter plot cuda_visible_devices exp_uid"
declare -a configs=(
    # "3 21 30.0 0.5 0.5 20 20 5000 False 1 10 50 1 True 0.0 1e-5 exp 1000 False $cuda_visible_devices $exp_uid"       # baseline
    "3 21 30.0 2 0.5 20 20 5000 False 1 10 50 1 True 0.0 1e-5 exp 1000 False $cuda_visible_devices $exp_uid"       # baseline
    "3 21 30.0 40 0.5 20 20 5000 False 1 10 50 1 True 0.0 1e-5 exp 1000 False $cuda_visible_devices $exp_uid"       # baseline

)

# Initialize pids array
declare -a pids=()

# Submit each job sequentially
for i in "${!configs[@]}"; do
    config=(${configs[$i]})
    
    # Run the experiment in parallel and capture the PID
    echo "----------------------------------------"
    echo "Starting experiment with:"
    echo "Dimension: ${config[0]}"
    echo "N: ${config[1]}"
    echo "L: ${config[2]}"
    echo "Beta: ${config[3]}"

    
    output=$(run_experiment "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "${config[7]}" "${config[8]}" "${config[9]}" "${config[10]}" "${config[11]}" "${config[12]}" "${config[13]}" "${config[14]}" "${config[15]}" "${config[16]}" "${config[17]}" "${config[18]}" "${config[19]}" "${config[20]}")
    # Store the PID of the background process
    readarray -t lines <<< "$output"
    # First line is the GPU index
    gpu_index="${lines[0]}"
    # Last line is the PID
    pid="${lines[-1]}"
    echo "GPU index: $gpu_index"
    echo "PID: $pid"
    echo "----------------------------------------"

    pids+=($pid)
    
    # Wait for 20 seconds before starting the next experiment
    sleep 30

done

echo "All experiments launched. Waiting for completion..."

# Print all PIDs of running experiments
echo "Running experiments with PIDs:"
for pid in "${pids[@]}"; do
    echo "  - $pid"
done

# Wait for all experiments to complete
all_done=false
while [ "$all_done" = false ]; do
    all_done=true
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null; then
            all_done=false
            echo "Process $pid is still running..."
            break
        fi
    done
    
    if [ "$all_done" = false ]; then
        echo "Waiting for processes to complete..."
        sleep 30
    fi
done

echo "All jobs completed"
send_email "All DFT jobs completed" "All DFT jobs completed"

# Save the script and src folders to the output directory as a zip