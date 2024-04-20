from __future__ import division
import cvxpy as cvx
import math
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools
from networkx.algorithms.bipartite.matching import maximum_matching
import numpy as np
import networkx as nx

TOLERANCE = np.finfo(np.float64).eps * 10.

def compute_pi_expo_fair(
    rel_mat: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    n_doc = rel_mat.shape[1]
    n_query = 1
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    pi = cvx.Variable((n_doc, K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        basis_ = np.zeros((1, n_doc))
        basis_[0][d] = 1
        pi_d = basis_ @ pi
        obj += rel_mat[:, d] @ pi_d @ v
        basis_ = np.ones((n_doc, 1))
        constraints += [pi_d @ basis_ <= 1]

    for k in np.arange(K):
        basis_ = np.zeros((K, 1))
        basis_[k][0] = 1
        pi_d = pi @ basis_
        basis_ = np.ones((1, K))
        constraints += [basis_ @ pi_d <= 1]

    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]
    constraints += [f @ pi @ v == 0]
    prob = cvx.Problem(cvx.Maximize(obj), constraints)

    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_unfiar(
    rel_mat: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    n_doc = rel_mat.shape[1]
    n_query = 1
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    pi = cvx.Variable((n_doc, K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        basis_ = np.zeros((1, n_doc))
        basis_[0][d] = 1
        pi_d = basis_ @ pi
        obj += rel_mat[:, d] @ pi_d @ v
        basis_ = np.ones((n_doc, 1))
        constraints += [pi_d @ basis_ == 1]

    for k in np.arange(K):
        basis_ = np.zeros((K, 1))
        basis_[k][0] = 1
        pi_d = pi @ basis_
        basis_ = np.ones((1, K))
        constraints += [basis_ @ pi_d == 1]

    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]
    prob = cvx.Problem(cvx.Maximize(obj), constraints)

    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi

def calculate_score(U, G, P, v, u):
    score = pd.concat([U, G], axis=1)
    score['EXPO'] = P @ v
    dcg = np.round((u @ P @ v)[0][0], decimals=2)
    group_dtr = score.groupby('G')['EXPO'].mean()
    u_by_group = score.groupby('G')['U'].mean()
    dtr = np.round((group_dtr[0] * u_by_group[1]) / (group_dtr[1] * u_by_group[0]), decimals=2)
    score['CLK'] = (P @ v) * u.T
    group_dir = score.groupby('G')['CLK'].mean()
    dir = np.round((group_dir[0] * u_by_group[1]) / (group_dir[1] * u_by_group[0]), decimals=2)
    return dcg, dtr, dir

def show_rankings_1(U, G, nc,rs):
    colors = [(0.180, 0.180, 0.180), (0.667, 0.667, 0.667), (0.667, 0.667, 0.667)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    exp_df = pd.concat([U, G], axis=1)
    exp_df = exp_df.reset_index(drop=True)
    exp_df['V'] = 1 / np.log(exp_df.index + 2)
    exp_df['V'] = exp_df['V'].round(2)
    avg_u_by_group = exp_df.groupby('G')['U'].mean()
    exp_df['F_DT'] = exp_df['G'].apply(lambda x: (1 if x == 0 else -1) / (exp_df[exp_df['G'] == x].shape[0] * avg_u_by_group[x]))
    exp_df['F_DP'] = exp_df['G'].apply(lambda x: (1 if x == 0 else -1) / exp_df[exp_df['G'] == x].shape[0])
    exp_df['F_DI'] = exp_df['F_DT'] * exp_df['U']
    n_doc = U.shape[0]
    u = np.array(exp_df['U']).reshape((1, n_doc))
    v = np.array(exp_df['V']).reshape((n_doc, 1))
    f_di = np.array(exp_df['F_DI']).reshape((1, n_doc))
    f_dp = np.array(exp_df['F_DP']).reshape((1, n_doc))
    f_dt = np.array(exp_df['F_DT']).reshape((1, n_doc))

    optimal_P_uf = np.round(compute_pi_unfiar(u, v), decimals=2)
    optimal_P_DP = np.round(compute_pi_expo_fair(u, v, f_dp), decimals=2)
    optimal_P_DT = np.round(compute_pi_expo_fair(u, v, f_dt), decimals=2)
    optimal_P_DI = np.round(compute_pi_expo_fair(u, v, f_di), decimals=2)

    exp_df['EXPO_nf'] = optimal_P_uf[0] @ v
    exp_df['EXPO_DP'] = optimal_P_DP[0] @ v
    exp_df['EXPO_DT'] = optimal_P_DT[0] @ v
    exp_df['EXPO_DI'] = optimal_P_DI[0] @ v

    dcg_uf = np.round((u @ optimal_P_uf[0] @ v)[0][0], decimals=2)
    dcg_dt = np.round((u @ optimal_P_DT[0] @ v)[0][0], decimals=2)
    dcg_di = np.round((u @ optimal_P_DI[0] @ v)[0][0], decimals=2)
    dcg_dp = np.round((u @ optimal_P_DP[0] @ v)[0][0], decimals=2)

    avg_expo_by_group_dt_dtr = exp_df.groupby('G')['EXPO_DT'].mean()
    avg_expo_by_group_dp_dtr = exp_df.groupby('G')['EXPO_DP'].mean()
    avg_expo_by_group_di_dtr = exp_df.groupby('G')['EXPO_DI'].mean()
    avg_expo_by_group_uf_dtr = exp_df.groupby('G')['EXPO_nf'].mean()

    exp_df['CLK_nf'] = (optimal_P_uf[0] @ v) * u.T
    exp_df['CLK_DP'] = (optimal_P_DP[0] @ v) * u.T
    exp_df['CLK_DT'] = (optimal_P_DT[0] @ v) * u.T
    exp_df['CLK_DI'] = (optimal_P_DI[0] @ v) * u.T

    avg_clk_by_group_dt_dir = exp_df.groupby('G')['CLK_DT'].mean()
    avg_clk_by_group_dp_dir = exp_df.groupby('G')['CLK_DP'].mean()
    avg_clk_by_group_di_dir = exp_df.groupby('G')['CLK_DI'].mean()
    avg_clk_by_group_uf_dir = exp_df.groupby('G')['CLK_nf'].mean()

    dir_uf = np.round((avg_clk_by_group_uf_dir[0] * avg_u_by_group[1]) / (avg_clk_by_group_uf_dir[1] * avg_u_by_group[0]), decimals=2)
    dir_dt = np.round((avg_clk_by_group_dt_dir[0] * avg_u_by_group[1]) / (avg_clk_by_group_dt_dir[1] * avg_u_by_group[0]), decimals=2)
    dir_dp = np.round((avg_clk_by_group_dp_dir[0] * avg_u_by_group[1]) / (avg_clk_by_group_dp_dir[1] * avg_u_by_group[0]), decimals=2)
    dir_di = np.round((avg_clk_by_group_di_dir[0] * avg_u_by_group[1]) / (avg_clk_by_group_di_dir[1] * avg_u_by_group[0]), decimals=2)

    dtr_uf = np.round((avg_expo_by_group_uf_dtr[0] * avg_u_by_group[1]) / (avg_expo_by_group_uf_dtr[1] * avg_u_by_group[0]), decimals=2)
    dtr_dt = np.round((avg_expo_by_group_dt_dtr[0] * avg_u_by_group[1]) / (avg_expo_by_group_dt_dtr[1] * avg_u_by_group[0]), decimals=2)
    dtr_dp = np.round((avg_expo_by_group_dp_dtr[0] * avg_u_by_group[1]) / (avg_expo_by_group_dp_dtr[1] * avg_u_by_group[0]), decimals=2)
    dtr_di = np.round((avg_expo_by_group_di_dtr[0] * avg_u_by_group[1]) / (avg_expo_by_group_di_dtr[1] * avg_u_by_group[0]), decimals=2)

    total_number_graph = 4

    zipped_pairs_dp = birkhoff_von_neumann_decomposition(optimal_P_DP[0])
    coefficients_dp, permutations_dp = zip(*zipped_pairs_dp)
    coefficients_dp = np.array(coefficients_dp)
    permutations_dp = np.array(permutations_dp)
    total_number_graph += coefficients_dp.shape[0]

    zipped_pairs_dt = birkhoff_von_neumann_decomposition(optimal_P_DT[0])
    coefficients_dt, permutations_dt = zip(*zipped_pairs_dt)
    coefficients_dt = np.array(coefficients_dt)
    permutations_dt = np.array(permutations_dt)
    total_number_graph += coefficients_dt.shape[0]

    zipped_pairs_di = birkhoff_von_neumann_decomposition(optimal_P_DI[0])
    coefficients_di, permutations_di = zip(*zipped_pairs_di)
    coefficients_di = np.array(coefficients_di)
    permutations_di = np.array(permutations_di)
    total_number_graph += coefficients_di.shape[0]

    cof_uf = u @ (optimal_P_uf[0] - optimal_P_uf[0]) @ v
    cof_dp = u @ (optimal_P_uf[0] - optimal_P_DP[0]) @ v
    cof_dt = u @ (optimal_P_uf[0] - optimal_P_DT[0]) @ v
    cof_di = u @ (optimal_P_uf[0] - optimal_P_DI[0]) @ v

    strategies = ['UF','DP', 'DT', 'DI']

    values = [cof_uf[0][0],cof_dp[0][0], cof_dt[0][0], cof_di[0][0]]

    plt.figure(figsize=(8, 6),facecolor='#efefef')

    plt.bar(strategies, values, color='#2e2e2e')

    plt.xlabel('Strategies')
    plt.ylabel('Values')
    plt.title('Comparison of COF for Different Strategies')
    plt.title('Comparison of Ranking Strategies')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # Set background color
    plt.gca().set_facecolor('#efefef')
    plt.show()


    strategies = ['UF', 'DT', 'DI', 'DP']
    dcg_values = [dcg_uf, dcg_dt, dcg_di, dcg_dp]
    dir_values = [dir_uf, dir_dt, dir_di, dir_dp]
    dtr_values = [dtr_uf, dtr_dt, dtr_di, dtr_dp]
    colors = ['#2e2e2e', '#aaa9a9', 'dimgray']
    plt.figure(figsize=(10, 6),facecolor='#efefef')
    positions = np.arange(len(strategies))


    for i,dcg in enumerate(dcg_values):
        plt.barh(positions[i] - 0.2, dcg, height=0.2, color=colors[0], label='DCG')
    for i,dir_val in enumerate(dir_values):
        plt.barh(positions[i], dir_val, height=0.2, color=colors[1], label='DIR')
    for i,dtr in enumerate(dtr_values):
        plt.barh(positions[i] + 0.2, dtr, height=0.2, color=colors[2], label='DTR')

    for i, (dcg, dir_val, dtr) in enumerate(zip(dcg_values, dir_values, dtr_values)):
        plt.text(dcg + 0.05, i - 0.2, str(dcg), va='center', ha='left', color='black')
        plt.text(dir_val + 0.05, i, str(dir_val), va='center', ha='left', color='black')
        plt.text(dtr + 0.05, i + 0.2, str(dtr), va='center', ha='left', color='black')

    plt.yticks(positions, strategies)
    plt.xlabel('Values')
    plt.title('Comparison of Ranking Strategies')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Set background color
    plt.gca().set_facecolor('#efefef')
    # plt.gca().set_edgecolor('#efefef')

    # Show legends
    # plt.legend(['DCG', 'DIR', 'DTR'], loc='upper right')
    plt.show()


    nr = math.ceil(total_number_graph / (nc * 1.0))
    r = 0
    c = 0
    fig, axes = plt.subplots(nr, nc, figsize=(nr * rs, rs*nr),facecolor='#efefef')

    sns.heatmap(optimal_P_uf[0], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
    axes[r][c].set_title("UF")
    axes[r][c].set_xlabel(f'DCG {dcg_uf}  DTR {dtr_uf}  DIR {dir_uf} ')
    c += 1
    if(c==nc) :
      c = 0
      r += 1

    sns.heatmap(optimal_P_DP[0], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
    axes[r][c].set_title("DP")
    axes[r][c].set_xlabel(f'DCG {dcg_dp} COF_DP {cof_dp}')
    c += 1
    if(c==nc) :
      c = 0
      r += 1
    sns.heatmap(optimal_P_DT[0], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
    axes[r][c].set_title("DT")
    axes[r][c].set_xlabel(f'DCG {dcg_dt}  DTR {dtr_dt} COF_DT {cof_dt}')
    c += 1
    if(c==nc) :
      c = 0
      r += 1
    sns.heatmap(optimal_P_DI[0], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
    axes[r][c].set_title("DI")
    axes[r][c].set_xlabel(f'DCG {dcg_di}  DIR {dir_di} COF_DI {cof_di}')
    c+=1
    if(c==nc) :
      c = 0
      r += 1

    for i in range(coefficients_dp.shape[0]):
        sns.heatmap(permutations_dp[i], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
        axes[r][c].set_title(f"DP ranking_{i+1}")
        e1, e2, e3 = calculate_score(exp_df['U'], exp_df['G'], permutations_dp[i], v, u)
        axes[r][c].set_xlabel(f'Theta {np.round(coefficients_dp[i], decimals=2)}  DCG {e1}  DTR {e2}  DIR {e3}')
        c += 1
        if c == nc:
            r += 1
            c = 0

    for i in range(coefficients_dt.shape[0]):
        sns.heatmap(permutations_dt[i], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
        axes[r][c].set_title(f"DT ranking_{i+1}")
        e1, e2, e3 = calculate_score(exp_df['U'], exp_df['G'], permutations_dt[i], v, u)
        axes[r][c].set_xlabel(f'Theta {np.round(coefficients_dt[i], decimals=2)}  DCG {e1}  DTR {e2}  DIR {e3}')
        c += 1
        if c == nc:
            r += 1
            c = 0

    for i in range(coefficients_di.shape[0]):
        sns.heatmap(permutations_di[i], annot=True, cmap='Greys', vmin=0, vmax=1, ax=axes[r][c])
        axes[r][c].set_title(f"DI ranking_{i+1}")
        e1, e2, e3 = calculate_score(exp_df['U'], exp_df['G'], permutations_di[i], v, u)
        axes[r][c].set_xlabel(f'Theta {np.round(coefficients_di[i], decimals=2)}  DCG {e1}  DTR {e2}  DIR {e3}')
        c += 1
        if c == nc:
            r += 1
            c = 0
    plt.gca().set_facecolor('#efefef')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.show()
    return exp_df


def show_rankings_2(U,G):

  exp_df = pd.concat([U,G], axis=1)
  exp_df = exp_df.reset_index(drop=True)
  exp_df['V'] = 1 / np.log(exp_df.index + 2)
  exp_df['V'] = exp_df['V'].round(2)
  avg_u_by_group = exp_df.groupby('G')['U'].mean()
  exp_df['F_DT'] = exp_df['G'].apply(lambda x: (1 if x == 0 else -1) / (exp_df[exp_df['G'] == x].shape[0]*avg_u_by_group[x]))
  exp_df['F_DP'] = exp_df['G'].apply(lambda x: (1 if x == 0 else -1) / exp_df[exp_df['G'] == x].shape[0])
  exp_df['F_DI'] = exp_df['F_DT'] * exp_df['U']
  u = exp_df['U'].values
  v = exp_df['V'].values
  f_dt = exp_df['F_DT'].values
  f_dp = exp_df['F_DP'].values
  f_di = exp_df['F_DI'].values
  # Define the objective function
  def objective_function_uf(P_flat, u, v):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return -np.dot(np.dot(u, P), v)

  # Define the constraint: the sum of each row and each column of P must be 1
  def constraint_uf(P_flat):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return np.concatenate([np.sum(P, axis=1) - 1, np.sum(P, axis=0) - 1])

  # Initial guess for the elements of P
  initial_guess = np.zeros(u.shape[0] * v.shape[0])


  # Define the bounds for the elements of P
  bounds = [(0, 1) for _ in range(u.shape[0] * v.shape[0])]

  # Define the equality constraint
  equality_constraint_uf = {'type': 'eq', 'fun': constraint_uf}

  # Perform the optimization
  result_uf = minimize(objective_function_uf, initial_guess, args=(u, v), bounds=bounds, constraints=equality_constraint_uf)

  # Extract the optimal P and round to two decimal places
  optimal_P_uf = result_uf.x.reshape(u.shape[0], v.shape[0])
  initial_guess = np.zeros(u.shape[0] * v.shape[0])
  initial_guess[initial_guess == 0] = 0.5
  # Define the objective function
  def objective_function_D(P_flat, u, v):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return -np.dot(np.dot(u, P), v)

  # Define the constraint: each row and each column of P must sum to 1
  def constraint_D(P_flat):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return np.concatenate([np.sum(P, axis=1) - 1, np.sum(P, axis=0) - 1])

  # Define the new constraint: f^T . P . v = 0
  def new_constraint_DT(P_flat):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return np.dot(np.dot(f_dt, P), v)
  def new_constraint_DP(P_flat):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return np.dot(np.dot(f_dp, P), v)
  def new_constraint_DI(P_flat):
      P = P_flat.reshape(u.shape[0], v.shape[0])
      return np.dot(np.dot(f_di, P), v)

  # Combine all equality constraints
  equality_constraints_DT = [
      {'type': 'eq', 'fun': constraint_D},
      {'type': 'eq', 'fun': new_constraint_DT}
  ]
  equality_constraints_DP = [
      {'type': 'eq', 'fun': constraint_D},
      {'type': 'eq', 'fun': new_constraint_DP}
  ]
  equality_constraints_DI = [
      {'type': 'eq', 'fun': constraint_D},
      {'type': 'eq', 'fun': new_constraint_DI}
  ]
  # Perform the optimization
  result_DT = minimize(objective_function_D, initial_guess, args=(u, v), bounds=bounds,constraints=equality_constraints_DT)

  # Extract the optimal P and round to two decimal places
  optimal_P_DT = result_DT.x.reshape(u.shape[0], v.shape[0])
  initial_guess = np.zeros(u.shape[0] * v.shape[0])
  initial_guess[initial_guess == 0] = 0.7
  result_DP = minimize(objective_function_D, initial_guess, args=(u, v), bounds=bounds,constraints=equality_constraints_DP)

  # Extract the optimal P and round to two decimal places
  optimal_P_DP = result_DP.x.reshape(u.shape[0], v.shape[0])

  initial_guess = np.zeros(u.shape[0] * v.shape[0])
  initial_guess[initial_guess == 0] = 0.5
  result_DI = minimize(objective_function_D, initial_guess, args=(u, v), bounds=bounds,constraints=equality_constraints_DI)

  # Extract the optimal P and round to two decimal places
  optimal_P_DI = result_DI.x.reshape(u.shape[0], v.shape[0])

  fig, axes = plt.subplots(1, 4, figsize=(20, 5))

  sns.heatmap(optimal_P_uf, annot=True, cmap='YlGnBu', vmin=0, vmax=1,ax=axes[0])
  axes[0].set_title("UF")
  axes[0].set_xlabel(-result_uf.fun)

  sns.heatmap(optimal_P_DP, annot=True, cmap='YlGnBu', vmin=0, vmax=1,ax=axes[1])
  axes[1].set_title("DP")
  axes[1].set_xlabel(-result_DP.fun)

  sns.heatmap(optimal_P_DT, annot=True, cmap='YlGnBu', vmin=0, vmax=1,ax=axes[2])
  axes[2].set_title("DT")
  axes[2].set_xlabel(-result_DT.fun)


  sns.heatmap(optimal_P_DI, annot=True, cmap='YlGnBu', vmin=0, vmax=1,ax=axes[3])
  axes[3].set_title("DI")
  axes[3].set_xlabel(-result_DI.fun)

  # Show the plot
  plt.show()
# print("Maximum value of u^T * P * v:", -result.fun)
  return exp_df

def to_permutation_matrix(matches):

    n = len(matches)
    P = np.zeros((n, n))
    # This is a cleverer way of doing
    #
    #     for (u, v) in matches.items():
    #         P[u, v] = 1
    #
    P[tuple(zip(*(matches.items())))] = 1
    return P


def zeros(m, n):
    return np.zeros((m, n))


def hstack(left, right):
    return np.hstack((left, right))


def vstack(top, bottom):
    return np.vstack((top, bottom))


def four_blocks(topleft, topright, bottomleft, bottomright):
    return vstack(hstack(topleft, topright),
                  hstack(bottomleft, bottomright))


def to_bipartite_matrix(A):
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))


def to_pattern_matrix(D):
    result = np.zeros_like(D)
    # This is a cleverer way of doing
    #
    #     for (u, v) in zip(*(D.nonzero())):
    #         result[u, v] = 1
    #
    result[D.nonzero()] = 1
    return result


def birkhoff_von_neumann_decomposition(D):
    m, n = D.shape
    if m != n:
        raise ValueError('Input matrix must be square ({} x {})'.format(m, n))
    indices = list(itertools.product(range(m), range(n)))
    coefficients = []
    permutations = []

    S = D.astype('float64')
    while not np.all(S == 0):
        W = to_pattern_matrix(S)
        X = to_bipartite_matrix(W)

        G = nx.from_numpy_array(X)

        left_nodes = range(n)
        M = maximum_matching(G, left_nodes)

        M = {u: v % n for u, v in M.items() if u < n}
        P = to_permutation_matrix(M)

        q = min(S[i, j] for (i, j) in indices if P[i, j] == 1)
        coefficients.append(q)
        permutations.append(P)

        S -= q * P

        S[np.abs(S) < TOLERANCE] = 0.0
    return list(zip(coefficients, permutations))
