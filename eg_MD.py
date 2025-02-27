from src.models import *


Ns = (5, 5, 5)
N_vec = Ns[0]*Ns[1]*Ns[2]
Ls = (1, 1, 1)
beta = 0.5
max_iter = 1000
eval_iter = 1
mu = 1.5
objectives = np.zeros([6, max_iter//eval_iter])

ham = deterministicHamiltonian(Ns, Ls,beta)
H = np.zeros([N_vec, N_vec])
eye = np.eye(N_vec)
# H = jnp.array(H)
lr = 1e-2
iter = 1

while iter <= max_iter:

    gradient = ham.gradient(H)
    effective_lr = lr/(lr + beta * np.sqrt(iter))
    H = (1-effective_lr)*H + effective_lr*(gradient - mu*eye)
    objectives[:, iter-1] = np.array(ham.objective(H))
    if iter % eval_iter == 0:
        print(f"Iteration {iter}, Objective {objectives[0, iter-1]}")
    iter += 1



plt.figure(figsize=(12,5),dpi=100)
plt.subplot(1,2,1)
iters = np.linspace(1, max_iter, max_iter)
plt.plot(iters, objectives[0,:], label="objective")
plt.plot(iters, objectives[1,:], label='free energy')
plt.xlabel("iterations")
plt.ylabel("energy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(iters, objectives[2,:], label="kinetic energy")
plt.plot(iters, objectives[3,:], label="external energy")
plt.plot(iters, objectives[4,:], label="hartree energy")
plt.plot(iters, objectives[5,:], label="entropy")
plt.xlabel("iterations")
plt.ylabel("energy")

plt.legend()
plt.tight_layout()
plt.show()





