import numpy as np
import matplotlib.pyplot as plt


f, ax = plt.subplots(1,3)

def plot(i, E):
    idxs = [i for i in range(31)]
    ax[i].plot(idxs, [e[0,0] for e in E], c='b')
    ax[i].plot(idxs, [e[0,1] for e in E], c='r')
    ax[i].plot(idxs, [e[1,1] for e in E], c='g')

def compute_and_plot(E_finals : list):
    for i, E_final in enumerate(E_finals):
        _compute_and_plot(i, E_final)

def _compute_and_plot(i : int, E_final : np.ndarray):
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]])
    Q = np.array([[2,0],[0,0]])
    R = 10
    N = 30

    x = [np.zeros((1,2)) for _ in range(31)]
    E = [np.zeros((2,2)) for _ in range(31)]
    F = [np.zeros((2,2)) for _ in range(30)]
    u = [0 for _ in range(30)]

    E[N] = E_final
    x[0] = np.array([[100],[0]])

    for k in range(29,-1,-1):
        E[k] = Q + A.T @ E[k+1] @ A - A.T @ E[k+1] @ B @ np.linalg.inv(R + B.T @ E[k+1] @ B) @ B.T @ E[k+1] @ A
        F[k] = np.linalg.inv(R + B.T @ E[k+1] @ B) @ B.T @ E[k+1] @ A
        u[k] = -F[k] * x[k]
    plot(i, E)

def main():
    # J = x[N].T * S * x[N] + sum(k=0 : N-1) {x[k].T * Q * x[k] + u[k].T * R * u[k]}

    # 1. E[N] = [[1,0], [0,1]] : we care both final position and speed
    # 2. E[N] = [[20,0],[0,0]] : we care a lot about final position but not final speed
    # 3. E[N] = [[0,0],[0,0]] : we do not care at all about final state
    
    E_final_0 = np.array([[1,0],[0,1]])
    E_final_1 = np.array([[20,0],[0,0]])
    E_final_2 = np.array([[0,0],[0,0]])
    E_finals = [E_final_0, E_final_1, E_final_2]
    compute_and_plot(E_finals)
    plt.show()

if __name__=='__main__':
    main()






