export CUDA_VISIBLE_DEVICES=1
python -m mirrordft.trainings.train_SMD \
    --job_name='SMD' \
    --dim=1 \
    --N=101 \
    --L=10.0 \
    --beta=10 \
    --alpha=0.5 \
    --cheat=True \
    --N_samples=20 \
    --N_poles=20 \
    --max_iter=1000 \
    --raw=True \
    --ratio=1 \
    --eval_iter=10 \
    --update_poles_iter=50 \
    --lr=0.5 \
    --scf_compare=True \
    --mu=0.0 \
    --tol=1e-5 \
    --decay='exp' \
    --decay_iter=100 \
    --plot=False
