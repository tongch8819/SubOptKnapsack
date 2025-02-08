from application.base_task import BaseTask
from application.constraint import KnapsackConstraint

import numpy as np
from typing import Set, List


class MaximumFacilityLocationKnapsack(BaseTask):
    """
    The Maximum Facility Location Problem (Max-FLP) is a combinatorial optimization problem where the goal is to select a subset of facilities to maximize the total utility while satisfying certain constraints.

    The objective function is monotone submodular because:
    + Adding a facility never decreases the maximum assignment profit for any customer (monotonicity).
    + The marginal gain of adding a new facility decreases as more facilities are selected (submodularity).
    """

    def __init__(self, num_facilities, num_customers, profits = None, costs = None, budget = None):
        """
        Initializes the Maximum Facility Location Problem with a knapsack constraint.

        Args:
            num_facilities (int): Number of available facilities.
            num_customers (int): Number of customers.
            profits (ndarray): A (num_facilities x num_customers) matrix where profits[i][j]
                               is the benefit of assigning customer j to facility i.
            costs (list): A list of length num_facilities where costs[i] is the cost of opening facility i.
            budget (float): The total budget constraint on facility costs.
        """
        if profits is None or costs is None or budget is None:
            profits, costs, budget = generate_max_flp_instance(num_facilities, num_customers)
        assert profits.shape == (
            num_facilities, num_customers), "Profit matrix dimensions do not match."
        assert len(
            costs) == num_facilities, "Cost list length must match the number of facilities."

        self.num_facilities = num_facilities
        self.num_customers = num_customers
        

        self.profits = profits
        self.constraint = KnapsackConstraint(costs, budget=budget)

        self.is_mono = True  # Monotonicity

    def __repr__(self):
        return f"Max-FLP Instance: {self.num_facilities} facilities, {self.num_customers} customers, Budget={self.constraint.budget}"

    @property
    def ground_set(self):
        return np.arange(self.num_facilities)

    def objective(self, selected_facilities: List[int]):
        """
        Computes the objective value (total profit) for a given set of facilities.

        Args:
            selected_facilities (list): List of facility indices selected.

        Returns:
            float: The total profit of the selected facilities.
        """
        if sum(self.constraint.cost_func[i] for i in selected_facilities) > self.constraint.budget:
            return -np.inf  # Invalid selection due to budget violation
        if len(selected_facilities) == 0:
            return 0  # No facilities selected
        # Compute max profit for each customer considering only selected facilities
        if type(selected_facilities) == set:
            selected_facilities = list(selected_facilities)
        customer_profits = np.max(self.profits[selected_facilities, :], axis=0)
        return np.sum(customer_profits)

    def is_feasible(self, selected_facilities):
        """
        Checks if a given selection of facilities is within the budget.

        Args:
            selected_facilities (list): List of selected facility indices.

        Returns:
            bool: True if the selection is feasible, False otherwise.
        """
        return sum(self.constraint.cost_func[i] for i in selected_facilities) <= self.constraint.budget



def generate_max_flp_instance(num_facilities, num_customers, 
                              profit_range=(5, 15), profit_scale=1.0, profit_variability=0.5, 
                              cost_range=(10, 50), cost_scale=1.0, cost_variability=0.5, 
                              seed=None):
    """
    Generates a random Maximum Facility Location Problem instance.

    Args:
        num_facilities (int): Number of facilities.
        num_customers (int): Number of customers.
        profit_range (tuple): Min and max values for the base profit matrix.
        profit_scale (float): Scaling factor for profits.
        profit_variability (float): Controls how much profits vary per facility.
        cost_range (tuple): Min and max values for facility costs.
        cost_scale (float): Scaling factor for costs.
        cost_variability (float): Controls how much costs vary.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        profits (ndarray): Generated profit matrix (num_facilities × num_customers).
        costs (list): Generated facility costs (num_facilities).
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base profit matrix with uniform random values
    profits = np.random.uniform(low=profit_range[0], high=profit_range[1], 
                                size=(num_facilities, num_customers))

    # Introduce variability per facility (optional scaling by Gaussian noise)
    profits *= profit_scale * (1 + profit_variability * np.random.randn(num_facilities, 1))

    # Ensure profits remain positive
    profits = np.clip(profits, 0, None)

    # Generate facility costs
    costs = np.random.uniform(low=cost_range[0], high=cost_range[1], size=num_facilities)
    costs *= cost_scale * (1 + cost_variability * np.random.randn(num_facilities))
    
    # Ensure costs remain positive
    costs = np.clip(costs, 1, None)  

    budget_ratio = np.random.uniform(0.3, 0.7)
    budget = budget_ratio * sum(costs)

    return profits, costs.tolist(), budget


def main():
    # Example: 5 facilities, 4 customers
    profits = np.array([
        [8, 5, 3, 7],
        [6, 7, 4, 6],
        [7, 6, 8, 5],
        [5, 6, 7, 4],
        [4, 3, 5, 6]
    ])
    costs = [10, 12, 15, 8, 7]  # Facility opening costs
    budget = 25  # Budget constraint

    # Initialize problem instance
    model = MaximumFacilityLocationKnapsack(num_facilities=5, num_customers=4, profits=profits, costs=costs, budget=budget)

    # Test a facility selection
    S = [0, 3]  # Selecting facility 0 and 3
    print("Objective value:", model.objective(S))  # Compute total profit
    print("Is feasible:", model.is_feasible(S))  # Check feasibility



if __name__ == "__main__":
    main()
