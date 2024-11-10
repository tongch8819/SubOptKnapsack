from compute_knapsack_exp import model_factory

if __name__ == "__main__":
    task = "facebook"
    n = 500
    seed = 0
    budget = 0

    facebook = model_factory(task, n, seed, budget, cm = 'normal')
    print(facebook.A)

