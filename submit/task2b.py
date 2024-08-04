import random
import math
import json
import time
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix

class GeneticAlgorithm(ABC):
    """
    A class for solving problems using Genetic Algorithm

    Attributes
    --------
    pop_size: int
        Size of Population
    selection_size: int
        Size of selected population for further process
    num_gens: int
        Number of generations for evolution
    mutation_rate: float
        Probability for a mutation to occur
    elite_size: int
        Size of the number of elites for crossover
    crossover_rate: float
        Probability that crossover happens
    num_pts: int
        Number of points in a sequence (an individual)
    population: list[tuple]
        A list of current population where each of the individual contains its fitness score
    """
    def __init__(self, pop_size, selection_size, num_gens, mutation_rate,
                 elite_size, crossover_rate, 
                 num_pts):
        assert pop_size > elite_size
        self.pop_size = pop_size
        self.selection_size = selection_size
        self.num_gens = num_gens
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.num_pts= num_pts
        self.population = []

    @abstractmethod
    def crossover(self, parent1, parent2):
        """Perform cross over to produce offsprings.
        
        Parameters
        ------
        parent1: any
        parent2: any
        """
        pass

    @abstractmethod
    def mutate(self, individual):
        """Mutate an individual.
        
        Parameters
        ------
        individual: any
            The individual in a population
        """
        
        pass

    @abstractmethod
    def init_population(self):
        """Initialise the population.
        """
        pass
    
    @abstractmethod
    def calc_fitness(self, individual):
        """Abstract method to calculate fitness score of an individual
        
        Parameters
        ------
        individual: any
            The individual in a population
        """
        pass
    
    def rank_population(self,):
        """Calculate the fitness score of each individual, and sort the population in descending order
        
        Return
        ------
        Sorted population in list[tuple]
        """
        fit_results = [(individual, self.calc_fitness(individual)) for individual, _ in self.population]
        return sorted(fit_results, key=lambda x: x[1], reverse=True)
    
    def next_generation(self):
        """Evolves the existing generation and update the population.
        This functions first rank the individuals in the population, then performs
        crossover and mutation according to the probability set in prior.
        """
        
        ranked_population = self.rank_population()
        selected_population = ranked_population[:self.elite_size].copy()
        
        for _ in range(int((self.pop_size - self.elite_size) / 2)):
            
            if random.random() < self.crossover_rate:
                parent1 = sorted(
                    random.choices(ranked_population, k=self.selection_size),
                    key=lambda x: x[1], reverse=True
                )[0][0]
                parent2 = sorted(
                    random.choices(ranked_population, k=self.selection_size),
                    key=lambda x: x[1], reverse=True
                )[0][0]
                
                child1, child2 = self.crossover(parent1, parent2)
                
            else:#no crossover
                child1 = random.choices(selected_population)[0][0]
                child2 = random.choices(selected_population)[0][0]
                
            if random.random() < self.mutation_rate:
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
            
            selected_population.append((child1,self.calc_fitness(child1)))
            selected_population.append((child2,self.calc_fitness(child2)))
            
        self.population = selected_population
    
    def get_sol(self):
        """Return the individual that has the highest fitness score (serves as the current selected solution).

        Returns
        ----
        individual: any
            the individual that has the highest fitness score
        fitness: float
            fitness score of the individual
        """
        solution = self.rank_population()[0]
        individual = solution[0]
        fitness = solution[1]
        return individual, fitness

    def solve(self, early_stop: bool = False): #changing it to False so more generation can be done -Akmal
        """Entry point of the algorithm. Performs initialisation and generation evolution.

        Parameters
        --------
        early_stop: bool
            If set to true, algorithm ends if the last three generation has the highly similar best individuals (i.e. difference of the average with the existing is < 1e-4)
        """
        if len(self.population) == 0:
            self.init_population()
        solution_lists = []
        for i in range(self.num_gens):
            self.next_generation()
            if i%10 == 0:
                solution_lists.append(self.get_sol())
            #early stopping
            if early_stop and len(solution_lists) >= 3 and abs(solution_lists[-1][1] - np.mean([fitness for _, fitness in solution_lists[-3:]])) < 1e-6:
                print("Early termination. Reason: converged. ")
                break
            if len(solution_lists) > 10:
                solution_lists = solution_lists[-3:]

class MVPGeneticAlgorithm(GeneticAlgorithm):
    """
    A class for solving multiple vehicle problem using Genetic Algorithm

    Attributes
    --------
    car_types: list[dict]
        types of car that has different capacity and cost
    cars: list[dict]
        list of cars that carry the information of the id and the car_type
    custs: list[dict]
        list of customers that carry the information of id, current location (pt) and demand.
    dist_matrix: np.array
        
    pop_size: int
        Size of Population
    selection_size: int
        Size of selected population for further process
    num_gens: int
        Number of generations for evolution
    mutation_rate: float
        Probability for a mutation to occur
    elite_size: int
        Size of the number of elites for crossover
    crossover_rate: float
        Probability that crossover happens
    num_pts: int
        Number of points in a sequence (an individual)
    population: list[tuple]
        A list of current population where each of the individual contains its fitness score
    """
    def __init__(self, car_types, cars, custs, depot, *kwargs):
        self.car_types = car_types
        self.cars = cars
        self.custs = custs
        self.dist_matrix = self._generate_dist_matrix(depot, custs, len(cars))
        super().__init__(*kwargs)

    def _generate_dist_matrix(self, depot, custs, num_cars):
        """Internal function for generating distance matrix.
        The 2-D matrix is first generated according to the distance of all customers, assuming only one car is used (so it is treated as TSP first).
        Then, the matrix is expanding both horizontally and vertically, i.e., add num_cars of rows and cols which each row/column represents the distance for a car to travel from depot to a specific customer (and vice versa).
        An example individual is D - C1 - C2 - D - C3 - C4 - D. (Given 5 customers, 2 cars)
        In a numerical representation, it will be [0, 1, 2, 5, 3, 5, 6], where 5 and 6 represents the 1st and 2nd car respectively.
        This distance matrix serves as a lookup table, e.g. get the distance between 1 and 2 from [0, 1].

        Parameters
        --------
        depot: tuple
            original point of the cars
        custs: list[dict]
            list of customers that carry the information of id, current location (pt) and demand.
        num_cars: int
            number of cars
        """
        points = [cust['pt'] for cust in custs]
        dist_car = [math.dist(depot, point) * 100 for point in points]
        dist_matrix = distance_matrix(points, points) * 100
        
        # Expand the matrix
        out_arr = np.tile(np.array(dist_car)[:, np.newaxis], (1, num_cars))
        up_arr =np.hstack((dist_matrix,out_arr))
        low_arr=np.array([dist_car + [0] * num_cars]* num_cars)
        dist_matrix = np.vstack((up_arr, low_arr))
        return dist_matrix
    
    def init_population(self):
        """Initialise the population according to the preset population size.
        Each individual must start with "0" (indicating the depot point), and end with the last car id (which should be equal to number of points)
        This ensures that all the individual is a valid combination, starting from the first point to the point where the id is a car id, all the points belong to this car.
        Hence, the last point must be a car, and from second to the second last point can be either representing a car or a customer.
        
        """
        self.population = []
        for _ in range(self.pop_size):
            individual = np.hstack((np.array([0]),np.random.permutation([i for i in range(1, self.num_pts)]), np.array([self.num_pts])))
            self.population.append((individual, None))
        self.population = self.rank_population()
        
        
    def mutate(self, individual):
        """Mutate an individual.
        
        Parameters
        ------
        individual: np.array
            The individual in a population, it should be a sequence of customer/ car id.
        """
        for first_id in range(1, len(individual) - 1): #ensure not to mutate the first and last point to maintain validity
            if random.random() < 0.2: 
                sec_id = random.randint(1, len(individual) - 2)
                if sec_id != first_id:  #make sure doesnt swap with same element -Akmal
                    individual[first_id], individual[sec_id] = individual[sec_id], individual[first_id]
        return individual
    
    def crossover(self, parent1, parent2):
        """Crossover two parents to produce 2 children using order crossover.
        We follow typical textbook approach, but we do not crossover the first and the last id to maintain validity.

        Parameters
        ------
        parent1: np.array
            First parent.
        parent2: np.array
            Second parent.
        """
        # Step 0: Preserve the last id value, first id is 0 so no need to save.
        parent1_last_id = parent1[-1]
        parent2_last_id = parent2[-1]
        parent1 = parent1[1:-1].copy()
        parent2 = parent2[1:-1].copy()
        changable_node_nums = self.num_pts - 1
        #Step 1: Select 2 points randomly
        c1 = int(random.random() * changable_node_nums) + 1
        c2 = int(random.random() * changable_node_nums) + 1
        if c2 < c1:
            c1, c2 = c2, c1
        #Step 2: Swap position to form part of child 1 and child 2.
        C1 = [None for _ in parent2]
        C2 = [None for _ in parent1]
        C1[c1 - 1: c2] = parent2[c1 - 1: c2]
        C2[c1 - 1: c2] = parent1[c1 - 1: c2]

        #Step 3: Create lsit L1 and L2.
        L1 = np.hstack((parent1[c2:], parent1[:c2]))
        L2 = np.hstack((parent2[c2:], parent2[:c2]))
        L1_p = [i for i in L1 if i not in C1]
        L2_p = [i for i in L2 if i not in C2]
        
        # Step 4: Assign L1' and L2' to C1, C2.
        for i, j in enumerate(L1_p):
            C1[(c2+i)%len(C1)] = j 
        for i, j in enumerate(L2_p):
            C2[(c2+i)%len(C2)] = j 

        # Step 5: Recover the first and last id.
        C1 = np.hstack(([0], C1, [parent1_last_id]))
        C2 = np.hstack(([0], C2, [parent2_last_id]))
        return C1, C2
    
    def calc_fitness(self, individual, show_log=False):
        """Calculate the fitness score of an individual.
        The fitness score is 100 divided by the total cost of travel (of all the cars). i.e., the lower the cost the higher the score.

        Parameters
        ------
        individual: np.array
            An individual which is a sequence of id.
        show_log: bool
            If true, details of the individual will be printed.
        """
        invalid = False
        total_cost, total_distance = 0, 0
        assigned_cars = [-1 for _ in range(len(individual))] # 1 index
        cur_car_id = -1
        individual_cars = [{"cost": 0, "distance": 0, "demand": 0} for _ in self.cars]
        
        for i_r, j in enumerate(reversed(individual)):
            i = len(individual) - i_r - 1
            if j == 0: # Nothing to do with the first point
                continue
            if j > len(self.custs): # For car points
                cur_car_id = j
                weight = self.car_types[self.cars[cur_car_id - 1- len(self.custs)]['type']]['cost']
                distance = self.dist_matrix[individual[i-1] - 1, individual[i] - 1]
                assigned_cars[i] = cur_car_id
                total_cost += weight * distance
                total_distance += distance
                individual_cars [cur_car_id -1 - len(self.custs)]['distance'] += distance
                individual_cars [cur_car_id -1 - len(self.custs)]['cost'] += weight * distance
            else: # For customer points
                assert cur_car_id != -1
                weight = self.car_types[self.cars[cur_car_id -1- len(self.custs)]['type']]['cost']
                distance = self.dist_matrix[individual[i-1] - 1, individual[i] - 1]
                assigned_cars[i] = cur_car_id
                total_cost += weight * distance
                total_distance += distance
                individual_cars [cur_car_id -1 - len(self.custs)]['demand'] += self.custs[j -1]['demand']
                individual_cars [cur_car_id -1 - len(self.custs)]['distance'] += distance
                individual_cars [cur_car_id -1 - len(self.custs)]['cost'] += weight * distance
                if  individual_cars [cur_car_id -1 - len(self.custs)]['demand'] > self.car_types[self.cars[cur_car_id -1- len(self.custs)]['type']]['capacity']:
                    invalid = True # total demands of all the customers in this car exceed the maximum capacity, the individual is invalid hence fitness score is zero.
                    break
                
        if not invalid:
            total_demand = sum(car['demand'] for car in individual_cars) ### added weights -Akmal
            
            if total_cost > 0:
                cost_score = 100 / total_cost
            else:
                cost_score = 0

            demand_weight = 0.9
            cost_weight = 0.1
            fitness = (demand_weight * total_demand) - (cost_weight * total_cost)
            # fitness = 100/ total_cost
            if show_log:
              print("Total distance = {0:.3f} km".format(total_distance))
              print("Total cost = RM {0:.2f}".format(total_cost))
              print("Total demand = {}".format(total_demand))
              print()
              car_cnt = 0
              for i, j in enumerate(self.cars):
                if individual_cars[i]['distance'] == 0:
                    continue
                car_cnt+= 1
                print("Vehicle {0} (Type {1})".format(car_cnt, chr(j['type']+ord('A')))) #beautified counter
                print("Round Trip Distance: {0:.3f} km, Cost: RM {1:.2f}, Demand: {2}".format(individual_cars[i]['distance'],individual_cars[i]['cost'],individual_cars[i]['demand']))
                print("Depot -> ", end='')
                for ii, jj in enumerate(individual):
                    if assigned_cars[ii] != j['id']:
                        continue #ignore those not belong to me
                    elif assigned_cars[ii] == jj:
                        print("Depot ({0:.3f} km)".format(self.dist_matrix[individual[ii-1] - 1, individual[ii] - 1]))
                        break
                    print("C{0} ({1:.3f}km) -> ".format(jj, self.dist_matrix[individual[ii-1] - 1, individual[ii] - 1]), end='')
                print()
        else:
            if show_log:
                print("Infeasible.")
            fitness = 0
        return fitness

def gen_test_case(cust_size=30, rand_seed=42):
    """Generate test case.
    
    Parameters
    ---------
    cust_size: int
        Number of customers
    rand_seed: int
        random seed for random generation

    Returns
    ------
    car_types: list[dict]
        types of car
    depot: tuple
        starting point
    custs: list[dict]
        list of customers
    """
    car_types = [{"id": 0, "capacity": 25, "cost": 1.2},
                 {"id": 1, "capacity": 30, "cost": 1.5}]
    depot = (4.4184, 114.0932)
    custs = []
    random.seed(rand_seed)
    for i in range(cust_size):
          custs.append({"id": 11+i, "pt": (4.3032+random.random(), 113.8322+random.random()), "demand": random.randint(5,14) })

    return car_types, depot, custs

def save_test_case(out_dir, car_types, depot, custs):
    """Save test case to file in JSON.
    
    Parameters
    ---------
    out_dir: str
        file path of the JSON
    car_types: list[dict]
        types of car
    depot: tuple
        starting point
    custs: list[dict]
        list of customers
    """
    out_file = open(out_dir, "w")
    json.dump({"car_types": car_types,
                "depot": depot,
                "custs": custs
                },out_file,  indent = 4)
    out_file.close()


def load_problem(file_path='.'):
    """Load the problem.
    
    Parameters
    ---------
    file_path: str
        file path of the JSON

    Returns
    ------
    car_types: list[dict]
        types of car
    depot: tuple
        starting point
    custs: list[dict]
        list of customers
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['car_types'], data['depot'], data['custs']

for i in range(20, 60, 10):
    car_types, depot, custs = gen_test_case(i)
    save_test_case(f"MVP{i}.json",car_types, depot, custs)

sample_test = {"path": f"sample_data.json", "best_ind": [], "best_score": 0, "avg_score": 0, "avg_time": 0}
start_time = time.time()
car_types, depot, custs = load_problem(sample_test['path'])
num_cars = len(custs) * 2
cars = [{"id": len(custs) + i + 1, "type": i // int(num_cars / 2)} for i in range(num_cars)]
POP_SIZE = 10
SELECTION_SIZE = 4
ELITE_SIZE = 1
NUM_GENS = 2000
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9
NUM_PTS = num_cars + len(custs)

ga = MVPGeneticAlgorithm(
    car_types, cars, custs, depot,
    POP_SIZE, SELECTION_SIZE, NUM_GENS,
    MUTATION_RATE, ELITE_SIZE, CROSSOVER_RATE, NUM_PTS,
)

ga.solve()
best_individual, best_score = ga.get_sol()
end_time = time.time()
if best_score > sample_test['best_score']:
    sample_test['best_score'] = best_score
    sample_test['best_ind'] = best_individual
sample_test['avg_score'] += best_score
sample_test['avg_time'] += end_time - start_time

sample_test['avg_score'] /= 5
sample_test['avg_time'] /= 5

print(sample_test)


print("Final Result: ")
fitness_score = ga.calc_fitness(sample_test['best_ind'], show_log=True)
