import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


x = dict()
E = dict()
F = dict()
u = dict()

def plot(x, E, u):
    f, ax = plt.subplots(4,3,figsize=(14,12))
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['lines.linestyle'] = 'dashdot'

    plt.subplots_adjust(left=0.05,bottom=0.01,right=0.9,top=0.88,wspace=0.5,hspace=0.4)

    idxs_0 = [_i for _i in range(30)]
    idxs = [_i for _i in range(31)]
    suptitle = r'$x_{init} = [100,0]^T$'+'for all three cases where each column represents each case'+'\n'
    #suptitle += r'\ncase 1 : \begin{equation*}A=\begin{pmatrix}1&0&0&1\end{pmatrix}\end{equation*}'
    suptitle += r'case 1 : $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$'+'\t'
    suptitle += r'case 2 : $A = \begin{bmatrix} 20 & 0 \\ 0 & 0 \end{bmatrix}$'+'\t'
    suptitle += r'case 3 : $A = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$'

    f.suptitle(suptitle,fontsize=14)
    
    for j in range(3):
        key = str(j)
        ax[0,j].plot(idxs, [_x[0,0] for _x in x[key]], c='b')
        ax[0,j].plot(idxs, [_x[1,0] for _x in x[key]], c='r')
        ax[0,j].legend((r'position in $x$ axis - $[m]$',r'speed in $x$ axis - $[m/s]$'))
        ax[0,j].set_title('position and speed for the time horizon N',fontsize=10)
        ax[0,j].set_xlabel(r'each discrete time step $dt - [unit]$',fontsize=8)
        ax[0,j].set_ylabel(r'position $x - [m]$ | speed  $v - [m/s]$',fontsize=8)
 
        ax[1,j].plot(idxs, [_e[0,0] for _e in E[key]], c='b')
        ax[1,j].plot(idxs, [_e[0,1] for _e in E[key]], c='r')
        ax[1,j].plot(idxs, [_e[1,1] for _e in E[key]], c='g')
        ax[1,j].legend((r'$E[k](0,0)$',r'$E[k](0,1)$',r'$E[k](1,1)$'))
        ax[1,j].set_title(r'cost matrix $E[k]$ for the time horizon N',fontsize=10)
        ax[1,j].set_xlabel(r'each discrete time step $dt - [unit]$',fontsize=8)

        ax[2,j].plot(idxs_0, [_f[0,0] for _f in F[key]], c='b')
        ax[2,j].plot(idxs_0, [_f[0,1] for _f in F[key]], c='r')
        ax[2,j].legend((r'$F[k](0,0)$',r'$F[k](0,1)$'))
        ax[2,j].set_title(r'LQR gain matrix $F[k]$ for the time horizon N',fontsize=10)
        ax[2,j].set_xlabel(r'each discrete time step $dt - [unit]$',fontsize=8)

        ax[3,j].plot(idxs_0, [_u[0] for _u in u[key]], c='b')
        ax[3,j].set_title(r'control input $u[k]$ for the time horizon N',fontsize=10)
        ax[3,j].set_xlabel(r'each discrete time step $dt - [unit]$',fontsize=8)
        ax[3,j].set_ylabel(r'input acceleration  $v - [m^2/s]$',fontsize=8)

    plt.savefig('lqr_boundary_conditions_analysis_0.png')

def compute(x_init: np.ndarray, E_finals : list):
    for j, E_final in enumerate(E_finals):
        _compute(x_init, j, E_final)

def _compute(x_init: np.ndarray, j : int, E_final : np.ndarray):
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]])
    Q = np.array([[2,0],[0,0]])
    R = 10
    N = 30

    key = str(j)
    x[key] = [np.zeros((2,1)) for _ in range(31)]
    E[key] = [np.zeros((2,2)) for _ in range(31)]
    F[key] = [np.zeros((2,2)) for _ in range(30)]
    u[key] = [np.zeros((2,1)) for _ in range(30)]

    x[key][0] = np.array([[100],[0]])
    E[key][N] = E_final
    for k in range(N-1,-1,-1):
        E[key][k] = Q + A.T @ E[key][k+1] @ A - A.T @ E[key][k+1] @ B @ np.linalg.inv(R + B.T @ E[key][k+1] @ B) @ B.T @ E[key][k+1] @ A
        F[key][k] = 1/(R + B.T @ E[key][k+1] @ B) @ B.T @ E[key][k+1] @ A
    for k in range(N):
        u[key][k] = -F[key][k] @ x[key][k]
        x[key][k+1] = A @ x[key][k] + B @ u[key][k]

def main():
    # J = x[N].T * S * x[N] + sum(k=0 : N-1) {x[k].T * Q * x[k] + u[k].T * R * u[k]}

    # 1. E[N] = [[1,0], [0,1]] : we care both final position and speed
    # 2. E[N] = [[20,0],[0,0]] : we care a lot about final position but not final speed
    # 3. E[N] = [[0,0],[0,0]] : we do not care at all about final state
    x_init = np.array([[100],[0]])
    E_final_0 = np.array([[1,0],[0,1]])
    E_final_1 = np.array([[20,0],[0,0]])
    E_final_2 = np.array([[0,0],[0,0]])
    E_finals = [E_final_0, E_final_1, E_final_2]
    compute(x_init, E_finals)
    plot(x, E, u)
    plt.show()


if __name__=='__main__':
    main()






