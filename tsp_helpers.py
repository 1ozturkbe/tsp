"""Model building, subtour detection and plotting for the TSP row-generation notebook."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

from helpers import PALETTE


# ── Master problem ────────────────────────────────────────────────────────────

def build_master_problem(
    N: int,
    D: np.ndarray,
    EDGES: list,
    sec_cuts: list[frozenset],
) -> pyo.ConcreteModel:
    """
    Build the undirected TSP master MIP with the current set of SEC cuts.

    Parameters
    ----------
    N        : number of cities
    D        : N×N symmetric distance matrix
    EDGES    : list of (i,j) pairs with i < j  (undirected edges)
    sec_cuts : list of frozensets, each defining an SEC

    Returns
    -------
    model : Pyomo ConcreteModel ready to solve
    """
    model = pyo.ConcreteModel(name="TSP_Master")

    # ── Decision variables — one binary per undirected edge ───────────────────
    model.x = pyo.Var(EDGES, domain=pyo.Binary)

    # Helper: cost of edge (i,j) stored with i < j
    def edge_cost(i, j):
        return D[i, j]  # symmetric, so D[i,j] == D[j,i]

    # ── Objective: minimize total tour distance ───────────────────────────────
    model.obj = pyo.Objective(
        expr=sum(edge_cost(i, j) * model.x[i, j] for (i, j) in EDGES),
        sense=pyo.minimize,
    )

    # ── Degree constraint — each node must have exactly 2 active edges ────────
    def edge_var(i, j):
        """Return the model variable for the undirected edge between i and j."""
        return model.x[min(i, j), max(i, j)]

    model.degree = pyo.ConstraintList()
    for i in range(N):
        model.degree.add(
            sum(edge_var(i, j) for j in range(N) if j != i) == 2
        )

    # ── Subtour elimination constraints (lazy) ────────────────────────────────
    model.sec = pyo.ConstraintList()
    for S in sec_cuts:
        model.sec.add(
            sum(model.x[i, j] for (i, j) in EDGES if i in S and j in S)
            <= len(S) - 1
        )

    return model


# ── Subtour detection ─────────────────────────────────────────────────────────

def extract_solution(model: pyo.ConcreteModel, EDGES: list) -> dict:
    """Return a dict (i,j) → 0/1 from the solved Pyomo model (i < j for all keys)."""
    return {(i, j): round(pyo.value(model.x[i, j])) for (i, j) in EDGES}


def _adjacency(x_vals: dict, N: int) -> dict[int, list[int]]:
    """Undirected adjacency list of the active edges in `x_vals`."""
    adj: dict[int, list[int]] = {i: [] for i in range(N)}
    for (i, j), val in x_vals.items():
        if val > 0.5:
            adj[i].append(j)
            adj[j].append(i)
    return adj


def _trace_cycle(adj: dict[int, list[int]], start: int) -> list[int]:
    """
    Walk the degree-2 cycle containing `start`, always moving to the
    neighbor we didn't come from. Returns nodes in traversal order.
    """
    cycle = [start]
    prev, cur = start, adj[start][0]   # pick an arbitrary first step
    while cur != start:
        cycle.append(cur)
        nxt = next(nb for nb in adj[cur] if nb != prev)
        prev, cur = cur, nxt
    return cycle


def find_subtours(x_vals: dict, N: int) -> list[list[int]]:
    """
    Decompose an undirected binary solution into cycles (subtours).

    Each active edge (i,j) contributes to both i's and j's adjacency list.
    Since every node has degree exactly 2, the graph is a disjoint union of
    simple cycles. We trace each cycle by walking the adjacency list without
    retracing the edge we arrived on.

    Parameters
    ----------
    x_vals : dict (i,j) → 0/1, with i < j for all keys
    N      : number of nodes

    Returns
    -------
    subtours : list of lists; each sub-list is one cycle in traversal order
    """
    adj = _adjacency(x_vals, N)

    visited: set[int] = set()
    subtours: list[list[int]] = []

    for start in range(N):
        if start in visited:
            continue
        cycle = _trace_cycle(adj, start)
        visited.update(cycle)
        subtours.append(cycle)

    return subtours


def tour_length(x_vals: dict, D: np.ndarray) -> float:
    """Compute the total distance of a solution (sum over active edges)."""
    return sum(D[i, j] * v for (i, j), v in x_vals.items() if v > 0.5)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_instance(coords: np.ndarray) -> None:
    """Scatter the cities of a TSP instance with their indices."""
    N = len(coords)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=50, zorder=3, color='steelblue',
               edgecolors='white', linewidths=1.2)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), textcoords='offset points', xytext=(6, 4),
                    fontsize=10, fontweight='bold')
    ax.set_title(f"{N}-city TSP instance", fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_iteration(
    coords: np.ndarray,
    x_vals: dict,
    subtours: list[list[int]],
    iteration: int,
    obj_val: float,
    n_cuts: int,
    ax: plt.Axes | None = None,
) -> None:
    """
    Plot one row-generation iteration (undirected edges).

    Each subtour is drawn in a distinct color as a closed polygon.
    If the solution is a single tour the title will say so.

    Parameters
    ----------
    ax : optional existing Axes to draw into (used for grid layout).
         If None, a new figure is created and shown immediately.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))

    is_hamiltonian = len(subtours) == 1 and len(subtours[0]) == len(coords)

    for s_idx, subtour in enumerate(subtours):
        color = PALETTE[s_idx % len(PALETTE)]
        # Draw undirected edges as lines (closed cycle)
        cycle_coords = coords[subtour + [subtour[0]]]   # close the loop
        ax.plot(cycle_coords[:, 0], cycle_coords[:, 1],
                '-', color=color, lw=2.0, alpha=0.85, zorder=2)
        # Draw nodes
        ax.scatter(coords[subtour, 0], coords[subtour, 1],
                   s=160, color=color, zorder=3, edgecolors='white', linewidths=1.2)
        for node in subtour:
            ax.annotate(
                str(node),
                coords[node],
                textcoords='offset points',
                xytext=(6, 4),
                fontsize=9,
                fontweight='bold',
                color=color,
            )

    # Legend patch per subtour
    if not is_hamiltonian:
        patches = [
            mpatches.Patch(
                color=PALETTE[s % len(PALETTE)],
                label=f"Subtour {s+1}: (len={len(subtour)})",
            )
            for s, subtour in enumerate(subtours)
        ]
        ax.legend(handles=patches, fontsize=7, loc='upper left',
                  framealpha=0.9, title="Detected subtours")

    status = ("✓ Hamiltonian" if is_hamiltonian
              else f"{len(subtours)} subtours")
    ax.set_title(
        f"Iter {iteration}  |  Obj={obj_val:.2f}  |  SECs={n_cuts}\n{status}",
        fontsize=9,
    )
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.25)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_final_tour(
    coords: np.ndarray,
    x_vals: dict,
    obj_val: float,
    n_iterations: int,
    elapsed: float,
) -> None:
    """Draw the optimal undirected tour."""
    fig, ax = plt.subplots(figsize=(7, 6))
    color = 'royalblue'

    # Reconstruct tour order from undirected adjacency
    order = _trace_cycle(_adjacency(x_vals, len(coords)), 0)
    order.append(0)   # close the loop

    cx = coords[order, 0]
    cy = coords[order, 1]
    ax.plot(cx, cy, '-', color=color, lw=2.0, alpha=0.7, zorder=1)

    ax.scatter(coords[:, 0], coords[:, 1], s=160, color=color, zorder=3,
               edgecolors='white', linewidths=1.2)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=9, fontweight='bold')

    ax.set_title(
        f"Optimal TSP Tour  |  Total distance = {obj_val:.2f}\n"
        f"Solved in {n_iterations} iteration(s)  |  Wall time: {elapsed:.2f}s",
        fontsize=12,
    )
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_convergence(history: list[dict]) -> None:
    """Objective, number of subtours and cumulative SECs across iterations."""
    iters      = [h["iteration"]   for h in history]
    objs       = [h["obj"]         for h in history]
    n_subtours = [h["n_subtours"]  for h in history]
    n_cuts     = [h["n_cuts"] + (h["n_subtours"] if h["n_subtours"] > 1 else 0)
                  for h in history]   # total cuts after this iteration

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Objective ─────────────────────────────────────────────────────────────
    axes[0].plot(iters, objs, 'o-', color='steelblue', lw=2, ms=6)
    axes[0].axhline(objs[-1], color='salmon', ls='--', lw=1.5, label=f'Optimal = {objs[-1]:.2f}')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Objective value')
    axes[0].set_title('Objective vs Iteration')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # ── Number of subtours ────────────────────────────────────────────────────
    axes[1].bar(iters, n_subtours, color='coral', edgecolor='white', width=0.6)
    axes[1].axhline(1, color='steelblue', ls='--', lw=1.5, label='Target: 1 tour')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Number of subtours')
    axes[1].set_title('Subtours Detected vs Iteration')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')

    # ── Cumulative SECs ───────────────────────────────────────────────────────
    axes[2].step(iters, n_cuts, where='post', color='mediumpurple', lw=2)
    axes[2].fill_between(iters, n_cuts, step='post', alpha=0.25, color='mediumpurple')
    axes[2].set_xlabel('Iteration'); axes[2].set_ylabel('Cumulative SECs')
    axes[2].set_title('SECs Added vs Iteration')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Row Generation Convergence', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
