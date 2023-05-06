import random


# create a TSPGA class
# this class will solve the TSP problem using a genetic algorithm
class TSPGA:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.city_names = list(adjacency_matrix.keys())

    def create_chromosome(self):
        chromosome = random.sample(self.city_names, len(self.city_names))
        return chromosome + [chromosome[0]]

    def create_population(self, size):
        return [self.create_chromosome() for _ in range(size)]

    def fitness_function(self, chromosome):
        fitness = 0
        for i in range(len(chromosome) - 1):
            fitness += self.adjacency_matrix[chromosome[i]][chromosome[i + 1]]
        return fitness

    def fix_chromosome(self, chromosome):
        chromosome = list(set(chromosome))
        missing_cities = [city for city in self.city_names if city not in chromosome]
        random.shuffle(missing_cities)
        chromosome += missing_cities
        chromosome += [chromosome[0]]
        return chromosome

    def crossover(self, chromosome1, chromosome2, crossover_rate=0.7):
        if random.random() > crossover_rate:
            return chromosome1, chromosome2
        start = random.randint(0, len(chromosome1) - 1)
        end = random.randint(start, len(chromosome1) - 1)
        new_chromosome1 = (
            chromosome1[:start] + chromosome2[start:end] + chromosome1[end:]
        )
        new_chromosome2 = (
            chromosome2[:start] + chromosome1[start:end] + chromosome2[end:]
        )
        return self.fix_chromosome(new_chromosome1), self.fix_chromosome(
            new_chromosome2
        )

    def mutate(self, chromosome, mutation_rate=0.1):
        if random.random() > mutation_rate:
            return chromosome
        start = random.randint(0, len(chromosome) - 1)
        end = random.randint(start, len(chromosome) - 1)
        new_chromosome = (
            chromosome[:start] + chromosome[start:end][::-1] + chromosome[end:]
        )
        return self.fix_chromosome(new_chromosome)

    def evaluate(self, population):
        return sorted(population, key=self.fitness_function)

    def selection(self, population, elite_size=0.2):
        return population[: int(len(population) * elite_size)]

    def solve(self, population_size=100, generations=1000):
        population = self.create_population(population_size)
        for _ in range(generations):
            population = self.evaluate(population)
            selected = self.selection(population)
            population = selected + [
                self.mutate(self.crossover(*random.sample(selected, 2))[0])
                for _ in range(population_size - len(selected))
            ]
        return population[0]


def array_to_adjacency_matrix(array):
    adjacency_matrix = {}
    for i in range(len(array)):
        adjacency_matrix[str(i)] = {}
        for j in range(len(array)):
            adjacency_matrix[str(i)][str(j)] = array[i][j]
    return adjacency_matrix


if __name__ == "__main__":
    adjacency_matrix = {
        "A": {"A": 0, "B": 10, "C": 15, "D": 20},
        "B": {"A": 10, "B": 0, "C": 35, "D": 25},
        "C": {"A": 15, "B": 35, "C": 0, "D": 30},
        "D": {"A": 20, "B": 25, "C": 30, "D": 0},
    }

    opt = TSPGA(adjacency_matrix)
    waypoints = opt.solve()
    cost = opt.fitness_function(waypoints)
    print("Waypoints:", waypoints)
    print("Cost:", cost)
