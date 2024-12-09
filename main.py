import numpy as np

class ACO_BPP:
    def __init__(self, num_items, num_bins, weights, num_ants, evaporation_rate, max_evals=10000):
        self.num_items = num_items
        self.num_bins = num_bins
        self.weights = weights
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.max_evals = max_evals
        self.pheromones = np.random.rand(self.num_items, self.num_bins) # Initialize pheromone matrix with random values between 0 and 1
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_evals = 0

    def fitness(self, solution):
        # Calculate the difference between the heaviest and lightest bin
        bin_weights = np.zeros(self.num_bins)
        for item, bin_index in enumerate(solution):
            bin_weights[bin_index] += self.weights[item]
        return max(bin_weights) - min(bin_weights)

    def generate_ant_solution(self):
        # Generate a solution by assigning items to bins probabilistically based on pheromones
        solution = []
        for item in range(self.num_items):
            pheromones = self.pheromones[item]
            if np.sum(pheromones) == 0:
                # Avoid division by zero if all pheromones are zero
                probabilities = np.ones(self.num_bins) / self.num_bins
            else:
                probabilities = pheromones / np.sum(pheromones)
            bin_choice = np.random.choice(self.num_bins, p=probabilities)
            solution.append(bin_choice)
        return solution

    def update_pheromones(self, ant_solutions, ant_fitnesses):
        # Update pheromones based on the solutions and their fitnesses
        # Evaporate pheromones
        self.pheromones *= self.evaporation_rate
        # Add pheromones for each ant's solution
        for solution, fitness in zip(ant_solutions, ant_fitnesses):
            if fitness == 0:
                pheromone_deposit = 100  # Avoid division by zero if fitness is perfect
            else:
                pheromone_deposit = 100 / fitness  # Calculate the pheromone deposit
            for item, bin_index in enumerate(solution):
                self.pheromones[item][bin_index] += pheromone_deposit

    def run(self):
        # Run the ACO algorithm until max_evals is reached
        while self.fitness_evals < self.max_evals:
            ant_solutions = []
            ant_fitnesses = []
            num_ants_this_round = min(self.num_ants, self.max_evals - self.fitness_evals) # Ensures that the algorithm doesn’t exceed the limit of maximum evaluations
            # Each ant builds a solution
            for _ in range(num_ants_this_round):
                solution = self.generate_ant_solution()
                fitness_value = self.fitness(solution)
                ant_solutions.append(solution)
                ant_fitnesses.append(fitness_value)
                self.fitness_evals += 1

                # Track the best solution
                if fitness_value < self.best_fitness:
                    self.best_fitness = fitness_value
                    self.best_solution = solution

            # Update pheromones based on ant solutions and fitnesses
            self.update_pheromones(ant_solutions, ant_fitnesses)

        return self.best_solution, self.best_fitness


def run_single_trial():
    # Set the parameters for the experiment

    np.random.seed(1) # set the number according to the trial you are running
    num_ants = 10 # number of paths (solutions) constructed at each iteration
    evaporation_rate = 0.90 # evaporation rate

    # BPP1
    num_items_bpp1 = 500
    num_bins_bpp1 = 10
    weights_bpp1 = np.arange(1, num_items_bpp1 + 1)  # weights 1 to 500

    # Run ACO on BPP1
    print("Running ACO on BPP1...")
    aco_bpp1 = ACO_BPP(
        num_items=num_items_bpp1,
        num_bins=num_bins_bpp1,
        weights=weights_bpp1,
        num_ants=num_ants,
        evaporation_rate=evaporation_rate
    )
    best_solution_bpp1, best_fitness_bpp1 = aco_bpp1.run()
    print(f"BPP1 - Best Fitness: {best_fitness_bpp1}\n")

    # BPP2
    num_items_bpp2 = 500
    num_bins_bpp2 = 50
    weights_bpp2 = (np.arange(1, num_items_bpp2 + 1) ** 2) / 2 # item weights = (i^2) / 2

    # Run ACO on BPP2
    print("Running ACO on BPP2...")
    aco_bpp2 = ACO_BPP(
        num_items=num_items_bpp2,
        num_bins=num_bins_bpp2,
        weights=weights_bpp2,
        num_ants=num_ants,
        evaporation_rate=evaporation_rate
    )
    best_solution_bpp2, best_fitness_bpp2 = aco_bpp2.run()
    print(f"BPP2 - Best Fitness: {best_fitness_bpp2}")

# Run the single trial for both BPP1 and BPP2
run_single_trial()