# TSP / VRP Notebooks

Two Jupyter notebooks demonstrating classic integer-programming decomposition techniques on routing problems, built with [Pyomo](https://www.pyomo.org/) and the [HiGHS](https://highs.dev/) solver.

## Notebooks

- **`tsp_row_generation.ipynb`** — Solves the Traveling Salesman Problem via **row generation**: iteratively solves a relaxed MIP and adds subtour elimination constraints (SECs) until the solution forms a single Hamiltonian cycle.
- **`vrp_column_generation.ipynb`** — Solves the Capacitated Vehicle Routing Problem (CVRP) via **column generation**: iteratively solves a restricted master problem (route selection) and a pricing subproblem (route generation) until no negative reduced-cost route remains, then solves a final set-partitioning MIP for an integer solution.

Both notebooks include problem generation, solver setup, the optimization loop, and visualizations of each iteration plus convergence plots.

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run python -m ipykernel install --user --name tsp --display-name "Python (tsp)"
```

Then open either notebook in VS Code or Jupyter and select the **"Python (tsp)"** kernel.

## Dependencies

pyomo, highspy (HiGHS solver), numpy, matplotlib, networkx, ipykernel, jupyter.
