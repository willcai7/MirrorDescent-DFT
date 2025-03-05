beta_list=(0.1 0.3 1 3 10 30)
for beta in ${beta_list[@]}; do
    python src/trainings/train_SMD.py \
        --dim=1 \
        --N=101 \
        --L=20.0 \
        --beta=$beta \
        --alpha=0.5 \
        --N_electrons=2 \
        --cheat=True \
        --N_samples=10 \
        --N_poles=100 \
        --max_iter=2000 \
        --raw=False
done