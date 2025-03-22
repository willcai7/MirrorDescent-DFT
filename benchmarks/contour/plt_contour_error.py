import pickle
import matplotlib.pyplot as plt
import numpy as np
with open('./error_vs_beta.pkl', 'rb') as f:
    data = pickle.load(f)

betas = data['betas']
errors = data['errors']
N_poless = data['n_poless']

for dim in range(3):
    errors_dim = errors[dim]
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'legend.fontsize': 12
    })
    plt.figure(figsize=(6,5), dpi=300)
    for i in range(len(betas)):
        plt.semilogy(N_poless, errors_dim[i], label=r'$\beta$='+str(betas[i]), linewidth=2, color=plt.cm.autumn(i/len(betas)))
    plt.legend()
    plt.ylabel("Mean absolute error")
    plt.xlabel("Number of poles")
    plt.title(r"Error of contour method w.r.t $N_{poles}$ for "+str(dim+1)+"D")

    plt.legend(frameon=True, edgecolor='black', loc='best', bbox_to_anchor=None, fancybox=False, prop={'size': 9})
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2) 
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'contour_error_{dim+1}D.png', dpi=300)
    plt.close()

