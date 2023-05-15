import random
import pandas as pd
from tqdm import tqdm


# create a TSPGA class
# this class will solve the TSP problem using a genetic algorithm
class TSPGA:
    def __init__(self, distance_matrix, time_matrix, initial_point=0):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.nodes = list(distance_matrix.keys())
        self.initial_point = self.nodes[initial_point]
        self.chromosome_length = len(self.nodes) + 1
        self.best_score = float("inf")
        self.patience_count = 0

    def create_chromosome(self):
        node_without_initial_point = [
            node for node in self.nodes if node != self.initial_point
        ]
        chromosome = random.sample(
            node_without_initial_point, len(node_without_initial_point)
        )
        return [self.initial_point] + chromosome + [self.initial_point]

    def create_population(self, size):
        return [self.create_chromosome() for _ in range(size)]

    def fitness_function(self, chromosome, matrix):
        fitness_distance = 0
        for i in range(len(chromosome) - 1):
            fitness_distance += matrix[chromosome[i]][chromosome[i + 1]]
        return fitness_distance

    def crossover(self, chromosome1, chromosome2, crossover_rate=0.7):
        if random.random() > crossover_rate:
            return chromosome1, chromosome2
        chromosome_length = self.chromosome_length
        crossover_points = sorted(random.sample(range(1, chromosome_length), 2))

        # create offspring templates
        offspring1 = [None for _ in range(chromosome_length)]
        offspring2 = [None for _ in range(chromosome_length)]

        # set initial point
        offspring1[0], offspring1[-1] = self.initial_point, self.initial_point
        offspring2[0], offspring2[-1] = self.initial_point, self.initial_point

        # copy crossover segment
        offspring1[crossover_points[0] : crossover_points[1]] = chromosome2[
            crossover_points[0] : crossover_points[1]
        ]
        offspring2[crossover_points[0] : crossover_points[1]] = chromosome1[
            crossover_points[0] : crossover_points[1]
        ]

        # define what's not in the crossover segment
        not_offspring1 = [item for item in chromosome1 if item not in offspring1]
        not_offspring2 = [item for item in chromosome2 if item not in offspring2]

        # assert empty slots equal to the length of the not_offspring
        assert offspring1.count(None) == len(not_offspring1)
        assert offspring2.count(None) == len(not_offspring2)

        # fill in the remaining slots
        for i in range(1, chromosome_length - 1):
            if offspring1[i] is None:
                offspring1[i] = not_offspring1.pop(0)
            if offspring2[i] is None:
                offspring2[i] = not_offspring2.pop(0)

        return offspring1, offspring2

    def mutate(self, chromosome, mutation_rate=0.1):
        if random.random() > mutation_rate:
            return chromosome
        chromosome_length = self.chromosome_length
        inversion_points = sorted(random.sample(range(1, chromosome_length - 1), 2))
        chromosome[inversion_points[0] : inversion_points[1]] = chromosome[
            inversion_points[0] : inversion_points[1]
        ][::-1]

        return chromosome

    def non_dominated_sort(self, population, fitness_distance, fitness_time):
        fitness_distance = [
            self.fitness_function(chromosome, self.distance_matrix)
            for chromosome in population
        ]
        fitness_time = [
            self.fitness_function(chromosome, self.time_matrix)
            for chromosome in population
        ]
        dominance_count = [0 for _ in range(len(population))]

        domination_table = []
        for i in range(len(population)):
            domination_table.append(
                {
                    "population_index": i,
                    "fitness_distance": fitness_distance[i],
                    "fitness_time": fitness_time[i],
                    "dominance_count": dominance_count[i],
                }
            )

        fronts = []
        fronts_index = [None for _ in range(len(population))]
        front_index = 0
        while len(domination_table) > 0:
            for i in range(len(domination_table)):
                for j in range(len(domination_table)):
                    if i == j:
                        continue
                    if (
                        domination_table[i]["fitness_distance"]
                        <= domination_table[j]["fitness_distance"]
                        and domination_table[i]["fitness_time"]
                        <= domination_table[j]["fitness_time"]
                    ) and (
                        domination_table[i]["fitness_distance"]
                        < domination_table[j]["fitness_distance"]
                        or domination_table[i]["fitness_time"]
                        < domination_table[j]["fitness_time"]
                    ):
                        domination_table[j]["dominance_count"] += 1

            non_dominated = []
            for row in domination_table:
                if row["dominance_count"] == 0:  # type: ignore
                    non_dominated.append(row["population_index"])  # type: ignore
            for i in non_dominated:
                fronts_index[i] = front_index  # type: ignore
            fronts.append(non_dominated)
            domination_table = [
                row
                for row in domination_table
                if row["population_index"] not in non_dominated
            ]
            for row in domination_table:
                row["dominance_count"] = 0

            front_index += 1

        return fronts, fronts_index

    def crowding_distance(self, population, fronts, fitness_distance, fitness_time):
        fitness_distance = [
            self.fitness_function(chromosome, self.distance_matrix)
            for chromosome in population
        ]
        fitness_time = [
            self.fitness_function(chromosome, self.time_matrix)
            for chromosome in population
        ]

        population_crowding_distance = [0 for _ in range(len(population))]
        for front in fronts:
            for i in range(len(front)):
                for j in range(len(front)):
                    if i == j:
                        continue
                    population_crowding_distance[front[i]] += (
                        (fitness_distance[i] - fitness_distance[j]) ** 2
                        + (fitness_time[i] - fitness_time[j]) ** 2
                    ) ** 0.5

        return population_crowding_distance

    def evaluate(self, population):
        fitness_distance = [
            self.fitness_function(chromosome, self.distance_matrix)
            for chromosome in population
        ]
        fitness_time = [
            self.fitness_function(chromosome, self.time_matrix)
            for chromosome in population
        ]

        return fitness_distance, fitness_time

    def tournament(
        self, population, fronts_index, population_crowding_distance, size=0.2
    ):
        selected = []
        for _ in range(int(len(population) * size)):  # type: ignore
            i, j = random.sample(range(len(population)), 2)
            if fronts_index[i] < fronts_index[j]:
                selected.append(population[i])
            elif fronts_index[i] > fronts_index[j]:
                selected.append(population[j])
            elif population_crowding_distance[i] < population_crowding_distance[j]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        return selected

    def elitism(self, population, fronts_index):
        selected = []
        for i in range(len(population)):
            if fronts_index[i] == 0:
                selected.append(population[i])
        return selected

    def split_front(
        self, fronts_index, population_crowding_distance, population, size=0.2
    ):
        sorted_front = sorted(
            zip(fronts_index, population_crowding_distance, population),
            key=lambda x: (x[0], -x[1]),
        )
        sorted_front = sorted_front[: int(len(sorted_front) * size)]

        fronts_index = [item[0] for item in sorted_front]
        population_crowding_distance = [item[1] for item in sorted_front]
        population = [item[2] for item in sorted_front]

        return fronts_index, population_crowding_distance, population

    def early_stopping(self, score, patience=10):
        if self.best_score > score:
            self.best_score = score
            self.patience = 0
        else:
            self.patience += 1

        if self.patience > patience:
            return True
        else:
            return False

    def fit(
        self,
        population_size=500,
        generations=1000,
        crossover=0.8,
        mutation=0.2,
        selection=0.2,
    ):
        population = self.create_population(population_size)
        f_d, f_t = 0, 0
        for _ in tqdm(range(generations)):
            f_d, f_t = self.evaluate(population)

            # show best distance and time in tqdm
            tqdm.write(f"{s_to_hm(min(f_t))}, {m_to_km(min(f_d))}")

            fronts, fronts_index = self.non_dominated_sort(population, f_d, f_t)
            population_crowding_distance = self.crowding_distance(
                population, fronts, f_d, f_t
            )
            fronts_index, population_crowding_distance, population = self.split_front(
                fronts_index, population_crowding_distance, population
            )
            parents = self.tournament(
                population, fronts_index, population_crowding_distance, selection
            )
            offsprings = [
                self.mutate(
                    random.choice(
                        self.crossover(
                            *random.sample(parents, 2), crossover_rate=crossover
                        )
                    ),
                    mutation,
                )
                for _ in range(population_size - len(parents))
            ]
            population = parents + offsprings

        f_d, f_t = self.evaluate(population)
        fronts, fronts_index = self.non_dominated_sort(population, f_d, f_t)
        population_crowding_distance = self.crowding_distance(
            population, fronts, f_d, f_t
        )
        fronts_index, population_crowding_distance, population = self.split_front(
            fronts_index, population_crowding_distance, population, size=1
        )
        selected_population = []
        for i in list(set(fronts_index)):
            index = fronts_index.index(i)
            selected_population.append(population[index])
        f_d, f_t = self.evaluate(selected_population)

        return selected_population, f_d, f_t


def array_to_adjacency_matrix(array):
    adjacency_matrix = {}
    for i in range(len(array)):
        adjacency_matrix[str(i)] = {}
        for j in range(len(array)):
            adjacency_matrix[str(i)][str(j)] = array[i][j]
    return adjacency_matrix


def s_to_hm(seconds):
    return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60)}m"


def m_to_km(m):
    return f"{m / 1000:.3f}km"


if __name__ == "__main__":
    distance_matrix = array_to_adjacency_matrix(
        pd.read_csv("distance_matrix.csv", sep=";", index_col=0).values
    )
    time_matrix = array_to_adjacency_matrix(
        pd.read_csv("duration_matrix.csv", sep=";", index_col=0).values
    )

    opt = TSPGA(distance_matrix, time_matrix)
    waypoints, f_d, f_t = opt.fit(population_size=250, generations=2000)

    result = pd.DataFrame({"waypoints": waypoints, "distance": f_d, "time": f_t})
    result["average"] = result.mean(numeric_only=True, axis=1)
    result = result.sort_values("average")

    print(result)
