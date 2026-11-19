"""Route utilities, RMP / pricing models and plotting for the VRP column-generation notebook."""
import itertools

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

from helpers import PALETTE


# ── Routes ────────────────────────────────────────────────────────────────────

def route_cost(route: list[int], D: np.ndarray, depot: int = 0) -> float:
    """
    Total travel distance of a route (including depot legs).

    Path: depot -> route[0] -> ... -> route[-1] -> depot
    """
    if not route:
        return 0.0
    cost = D[depot, route[0]]
    for a, b in zip(route, route[1:]):
        cost += D[a, b]
    cost += D[route[-1], depot]
    return cost


def is_feasible(route: list[int], capacity: int) -> bool:
    """A route is feasible iff it visits at most `capacity` customers."""
    return len(route) <= capacity


# ── Restricted master problem ─────────────────────────────────────────────────

def build_rmp(
    routes:     list[list[int]],
    costs:      list[float],
    customers:  list[int],
) -> pyo.ConcreteModel:
    """
    Build the LP relaxation of the set-covering Restricted Master Problem.

    Parameters
    ----------
    routes    : list of route definitions (each a list of customer node indices)
    costs     : corresponding route costs
    customers : list of customer node indices that must be covered

    Returns
    -------
    Pyomo ConcreteModel with a `dual` Suffix attached for dual-variable retrieval.
    """
    model = pyo.ConcreteModel(name="VRP_RMP")

    R = list(range(len(routes)))

    # ── Suffix to import dual variables after LP solve ────────────────────────
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ── Decision variables: route selection weights (LP relaxation) ───────────
    model.lam = pyo.Var(R, domain=pyo.NonNegativeReals)

    # ── Objective: minimize total route cost ──────────────────────────────────
    model.obj = pyo.Objective(
        expr=sum(costs[r] * model.lam[r] for r in R),
        sense=pyo.minimize,
    )

    # ── Coverage constraints: each customer covered at least once ─────────────
    # Keyed by customer index so duals are easy to look up
    def cover_rule(m, i):
        return sum(m.lam[r] for r in R if i in routes[r]) >= 1

    model.cover = pyo.Constraint(customers, rule=cover_rule)

    return model


def solve_rmp(model: pyo.ConcreteModel, solver) -> tuple[float, dict[int, float]]:
    """
    Solve the RMP LP and return (objective_value, {customer: dual_value}).

    The dual of each coverage constraint is the shadow price π_i.
    For a ≥ constraint in a minimization LP, appsi_highs returns non-negative
    duals directly: π_i ≥ 0 (credit for covering customer i).
    """
    result = solver.solve(model)
    obj_val = pyo.value(model.obj)

    # appsi_highs returns π_i ≥ 0 for ≥ constraints in a min LP — use directly
    duals = {}
    for i in model.cover:
        raw = model.dual.get(model.cover[i], 0.0)
        duals[i] = raw if raw is not None else 0.0

    return obj_val, duals


# ── Pricing subproblem ────────────────────────────────────────────────────────

def build_pricing_subproblem(
    D:         np.ndarray,
    duals:     dict[int, float],
    customers: list[int],
    depot:     int,
    capacity:  int,
) -> pyo.ConcreteModel:
    """
    Build the pricing subproblem MIP with UNDIRECTED edges.

    Two explicit binary variables handle the depot legs:
      x_out[j] = 1  →  route departs depot to customer j first
      x_in[j]  = 1  →  route returns to depot from customer j last

    For a singleton route, x_out[j]=x_in[j]=1, so the depot–j edge is
    traversed TWICE and costs 2·D[0,j] — the out-and-back distance.
    For a multi-customer route, x_out[j1]=1 and x_in[jk]=1 (j1≠jk).

    Subtour elimination among customers is handled by SECs for all
    customer subsets of size 3..Q (enumeration is feasible when Q is small).

    Parameters
    ----------
    D         : N×N symmetric Euclidean distance matrix
    duals     : {customer_index: π_i} dual variables from the current RMP solve
    customers : list of customer node indices
    depot     : depot node index (0)
    capacity  : vehicle capacity Q

    Returns
    -------
    Pyomo ConcreteModel ready to solve.
    """
    model = pyo.ConcreteModel(name="VRP_Pricing")

    CC_EDGES = [(i, j) for i in customers
                        for j in customers if i < j]     # undirected customer pairs

    # ── Decision variables ────────────────────────────────────────────────────
    # Depot legs: two separate binary variables so the singleton out-and-back
    # cost (2·D[0,j]) is represented unambiguously by x_out[j]+x_in[j]=2.
    model.x_out = pyo.Var(customers, domain=pyo.Binary)   # outbound depot→j
    model.x_in  = pyo.Var(customers, domain=pyo.Binary)   # inbound  j→depot
    # Undirected customer–customer edges
    model.x_c   = pyo.Var(CC_EDGES,  domain=pyo.Binary)
    # Visit indicator
    model.v     = pyo.Var(customers,  domain=pyo.Binary)

    # ── Degree helper ─────────────────────────────────────────────────────────
    def cust_degree(i):
        """Total undirected degree of customer i (both depot legs + CC edges)."""
        d = model.x_out[i] + model.x_in[i]
        for (a, b) in CC_EDGES:
            if a == i or b == i:
                d = d + model.x_c[a, b]
        return d

    # ── Objective: minimize reduced cost = travel cost − dual credits ─────────
    # Depot-leg cost counts x_out[j]+x_in[j] times D[0,j], so:
    #   singleton  (x_out[j]=x_in[j]=1)  → 2·D[0,j]  ✓
    #   endpoint   (x_out OR x_in = 1)   → 1·D[0,j]  ✓
    model.obj = pyo.Objective(
        expr=(
            sum(D[depot, j] * (model.x_out[j] + model.x_in[j]) for j in customers)
            + sum(D[i, j]   * model.x_c[i, j] for (i, j) in CC_EDGES)
            - sum(duals.get(i, 0.0) * model.v[i] for i in customers)
        ),
        sense=pyo.minimize,
    )

    # ── Depot: depart once and return once ────────────────────────────────────
    model.depot_out = pyo.Constraint(
        expr=sum(model.x_out[j] for j in customers) == 1
    )
    model.depot_in = pyo.Constraint(
        expr=sum(model.x_in[j]  for j in customers) == 1
    )

    # ── Customer degree: 2 if visited, 0 otherwise ────────────────────────────
    model.cust_degree_con = pyo.ConstraintList()
    for i in customers:
        model.cust_degree_con.add(cust_degree(i) == 2 * model.v[i])

    # ── Capacity: at most Q customers per route ───────────────────────────────
    model.capacity_con = pyo.Constraint(
        expr=sum(model.v[i] for i in customers) <= capacity
    )

    # ── Subtour elimination constraints (SECs) ────────────────────────────────
    # For every customer subset S of size 3..Q, the number of CC edges within S
    # is at most |S|-1, preventing isolated customer sub-cycles.
    # (Size-2 pure customer cycles are already ruled out by the depot constraints.)
    model.sec = pyo.ConstraintList()
    for size in range(3, capacity + 1):
        for S in itertools.combinations(customers, size):
            S_set = set(S)
            edges_in_S = [(a, b) for (a, b) in CC_EDGES
                          if a in S_set and b in S_set]
            if edges_in_S:
                model.sec.add(
                    sum(model.x_c[a, b] for (a, b) in edges_in_S) <= size - 1
                )

    return model


def extract_route_from_pricing(
    model:     pyo.ConcreteModel,
    depot:     int,
    customers: list[int],
) -> list[int]:
    """
    Reconstruct the ordered customer sequence from undirected edge variables.

    Uses x_out to identify the first customer (depot→ first), then traces
    the Hamiltonian path through active CC edges until reaching the last
    customer (whose x_in=1 returns to depot).
    Returns an ordered list of customer nodes (depot legs not included).
    """
    CC_EDGES = [(i, j) for i in customers for j in customers if i < j]

    # Identify the first and last customers from depot legs
    first = next((j for j in customers
                  if pyo.value(model.x_out[j]) is not None
                  and pyo.value(model.x_out[j]) > 0.5), None)
    last  = next((j for j in customers
                  if pyo.value(model.x_in[j])  is not None
                  and pyo.value(model.x_in[j])  > 0.5), None)

    if first is None:
        return []

    # Singleton route: same customer is both first and last
    if first == last:
        return [first]

    # Build undirected CC adjacency
    adj: dict[int, list[int]] = {}
    for (i, j) in CC_EDGES:
        val = pyo.value(model.x_c[i, j])
        if val is not None and val > 0.5:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)

    # Walk from first to last following CC edges
    route: list[int] = [first]
    prev, cur = depot, first    # treat depot as "where we came from"
    while cur != last:
        neighbors = adj.get(cur, [])
        nxt = next((nb for nb in neighbors if nb != prev), None)
        if nxt is None:
            break
        route.append(nxt)
        prev, cur = cur, nxt

    return route


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_instance(
    coords:    np.ndarray,
    depot:     int,
    customers: list[int],
    capacity:  int,
) -> None:
    """Scatter the depot and customers of a CVRP instance."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(coords[customers, 0], coords[customers, 1],
               s=120, color='steelblue', edgecolors='white', linewidths=1.2,
               zorder=3, label='Customers')
    ax.scatter(*coords[depot], s=250, color='crimson', marker='*', zorder=4,
               edgecolors='black', linewidths=0.8, label=f'Depot ({depot})')
    for i in customers:
        ax.annotate(str(i), coords[i], textcoords='offset points',
                    xytext=(5, 4), fontsize=9, fontweight='bold', color='navy')
    ax.annotate(str(depot), coords[depot], textcoords='offset points',
                xytext=(6, 4), fontsize=11, fontweight='bold', color='crimson')
    ax.set_title(f"CVRP Instance  |  {len(customers)} customers  |  Q = {capacity}",
                 fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _draw_route(ax, route, depot, coords, color, lw=1.8, alpha=0.75,
                ls='-', label=None, zorder=2):
    """Draw a single vehicle route (depot → customers → depot) as a polyline."""
    path = [depot] + route + [depot]
    xs = coords[path, 0]
    ys = coords[path, 1]
    ax.plot(xs, ys, ls=ls, color=color, lw=lw, alpha=alpha,
            zorder=zorder, label=label)


def plot_cg_iteration(
    coords:       np.ndarray,
    routes:       list[list[int]],
    lambdas:      list[float],
    new_route:    list[int] | None,
    new_rc:       float | None,
    iteration:    int,
    rmp_obj:      float,
    depot:        int,
    ax:           plt.Axes | None = None,
) -> None:
    """
    Plot one column generation iteration.

    Active RMP routes (λ > 0.01) are drawn with width proportional to λ.
    The newly added route from pricing is highlighted in red.

    Parameters
    ----------
    ax : if provided, draw into existing Axes; otherwise create & show a new figure.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))

    # ── Plot active RMP routes ────────────────────────────────────────────────
    active_r = [(r, lam) for r, lam in zip(routes, lambdas) if lam > 0.01]
    for idx, (route, lam) in enumerate(active_r):
        color = PALETTE[idx % len(PALETTE)]
        lw    = max(0.8, 4.0 * lam)   # width proportional to λ
        _draw_route(ax, route, depot, coords, color=color, lw=lw, alpha=0.7)
        # Mark customers
        ax.scatter(coords[route, 0], coords[route, 1],
                   s=60, color=color, zorder=3,
                   edgecolors='white', linewidths=0.8)

    # ── Highlight new route in bold red ──────────────────────────────────────
    if new_route:
        _draw_route(ax, new_route, depot, coords,
                    color='red', lw=2.5, alpha=0.95, ls='--',
                    label=f'New route  rc={new_rc:.3f}', zorder=5)
        ax.scatter(coords[new_route, 0], coords[new_route, 1],
                   s=100, color='red', zorder=6,
                   edgecolors='black', linewidths=0.8)
        for node in new_route:
            ax.annotate(str(node), coords[node],
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=7, color='darkred', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    # ── Depot marker ─────────────────────────────────────────────────────────
    ax.scatter(*coords[depot], s=200, color='crimson', marker='*',
               zorder=7, edgecolors='black', linewidths=0.8)

    n_active = len(active_r)
    ax.set_title(
        f"Iter {iteration}  |  RMP obj={rmp_obj:.2f}\n"
        f"Active routes (λ>0): {n_active}  |  Pool size: {len(routes)}",
        fontsize=8,
    )
    ax.set_xlabel('x', fontsize=7); ax.set_ylabel('y', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_final_solution(
    coords:    np.ndarray,
    D:         np.ndarray,
    routes:    list[list[int]],
    lambdas:   list[float],
    obj_val:   float,
    depot:     int,
    elapsed:   float,
) -> None:
    """Draw the final integer VRP solution."""
    fig, ax = plt.subplots(figsize=(8, 7))

    selected = [(r, int(round(lam))) for r, lam in zip(routes, lambdas) if lam > 0.5]
    patches  = []

    # All non-depot nodes (background scatter)
    others = [i for i in range(len(coords)) if i != depot]
    ax.scatter(coords[others, 0], coords[others, 1], s=80, color='lightgrey',
               zorder=1, edgecolors='white', linewidths=0.8)

    for v_idx, (route, _) in enumerate(selected):
        color = PALETTE[v_idx % len(PALETTE)]
        cost  = route_cost(route, D, depot)
        label = f"Vehicle {v_idx+1}: {route}  ({cost:.1f})"
        _draw_route(ax, route, depot, coords, color=color, lw=2.2,
                    alpha=0.85, label=label, zorder=2)
        ax.scatter(coords[route, 0], coords[route, 1],
                   s=120, color=color, zorder=3,
                   edgecolors='white', linewidths=1.0)
        for node in route:
            ax.annotate(str(node), coords[node],
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=8, fontweight='bold', color=color)
        patches.append(mpatches.Patch(color=color, label=label))

    # Depot
    ax.scatter(*coords[depot], s=300, color='crimson', marker='*',
               zorder=8, edgecolors='black', linewidths=1.0)
    ax.annotate('Depot', coords[depot], textcoords='offset points',
                xytext=(7, 5), fontsize=9, color='crimson', fontweight='bold')

    ax.set_title(
        f"Final Integer VRP Solution  |  Total distance = {obj_val:.2f}\n"
        f"Vehicles used: {len(selected)}  |  Wall time: {elapsed:.2f}s",
        fontsize=12,
    )
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(handles=patches, fontsize=7, loc='upper right',
              framealpha=0.9, title='Routes')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_convergence(history: list[dict], mip_obj: float, epsilon: float) -> None:
    """RMP LP objective, entering reduced cost and column-pool size across iterations."""
    iters      = [h["iteration"]    for h in history]
    rmp_objs   = [h["rmp_obj"]      for h in history]
    red_costs  = [h["reduced_cost"] for h in history]
    pool_sizes = [h["pool_size"] + (1 if h["reduced_cost"] < -epsilon else 0)
                  for h in history]   # pool size after this iteration's column add

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── RMP LP Objective ──────────────────────────────────────────────────────
    axes[0].plot(iters, rmp_objs, 'o-', color='steelblue', lw=2, ms=6)
    axes[0].axhline(mip_obj, color='salmon', ls='--', lw=1.5,
                    label=f'MIP optimal = {mip_obj:.2f}')
    axes[0].set_xlabel('CG Iteration'); axes[0].set_ylabel('LP Objective Value')
    axes[0].set_title('RMP LP Objective vs Iteration')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # ── Reduced cost of entering column ───────────────────────────────────────
    axes[1].plot(iters, red_costs, 's-', color='coral', lw=2, ms=6)
    axes[1].axhline(0, color='steelblue', ls='--', lw=1.5, label='rc = 0 (stop)')
    axes[1].fill_between(iters, red_costs, 0,
                         where=[rc < 0 for rc in red_costs],
                         alpha=0.25, color='coral', label='Improving region')
    axes[1].set_xlabel('CG Iteration'); axes[1].set_ylabel('Reduced Cost')
    axes[1].set_title('Entering Column Reduced Cost vs Iteration')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # ── Column pool growth ────────────────────────────────────────────────────
    axes[2].step(iters, pool_sizes, where='post', color='mediumpurple', lw=2)
    axes[2].fill_between(iters, pool_sizes, step='post',
                         alpha=0.25, color='mediumpurple')
    axes[2].scatter(iters, pool_sizes, s=40, color='mediumpurple', zorder=3)
    axes[2].set_xlabel('CG Iteration'); axes[2].set_ylabel('Columns in Pool')
    axes[2].set_title('Cumulative Column Pool Size vs Iteration')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Column Generation Convergence', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
