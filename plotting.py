import matplotlib
import numpy as np
import seaborn as sns
import pub_ready_plots as prp
from os.path import join as pjoin


def main():
    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    with prp.get_context(
        layout=prp.Layout.NEURIPS,  
        width_frac=1,              
        height_frac=0.2,   
        nrows=1,                   
        ncols=2                                
    ) as (fig, axs):
        
        font = {"size": 20, "weight": "bold", "family": "serif", "serif": ["Times New Roman"]}
        matplotlib.rc("font", **font)

        A = np.array(
                [[1, -0.5],
                [-0.5, 1]]
        )

        f = lambda x1, x2: x1**2 + x2**2 - 1 * x1 * x2
        fgrad = lambda x1, x2: np.array([2 * x1 - x2, 2*x2 - x1])

        xlimits = [-10, 10]
        ylimits = [-10, 10]
        numticks = 100

        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)

        zs = f(np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel()))
        Z = zs.reshape(X.shape)

        axs[0].contour(X, Y, Z, levels=30)

        xhat = np.array([8., 3.])

        for _ in range(10):
                g = -0.5 * fgrad(*xhat)
                axs[0].arrow(*xhat, *g, head_width=0.4, color="black")
                xhat += g

        axs[0].set_title(r"$\theta_{t+1} = \theta_t - \gamma \: \nabla_\theta \mathcal{L}_{\text{VFM}}(\theta)$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        xlimits = [-10, 10]
        ylimits = [-10, 10]
        numticks = 100
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = f(np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel()))
        Z = zs.reshape(X.shape)

        axs[1].contour(X, Y, Z, levels=30)

        xhat = np.array([8., 3.])

        for _ in range(10):
                g = -0.3 * np.linalg.inv(A) @ fgrad(*xhat)
                axs[1].arrow(*xhat, *g, head_width=0.4, color="black")
                xhat += g

        axs[1].set_title(r"$\theta_{t+1} = \theta_t - \gamma \: I_\varepsilon (\theta) ^{-1}\nabla_\theta \mathcal{L}_{\text{VFM}}(\theta)$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])

    fig.savefig(pjoin("results", "optimization.pdf"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()