import numpy as np

nQueens = 8
POPULATION = 100
TARGET_FIT = 28
MAX_ITER = 5000
MUTATION_RATE = 1
debug_f = False
debug_m = False


class createIndividual:
    def __init__(self):
        self.chromosome = None
        self.fitness = None
        self.probability = None
        
    def setChromosome(self, val):
        self.chromosome = val

    def setFitness(self, val):
        self.fitness = val

    def setProbability(self, val):
        self.probability = val

    def getAll(self):
        return {'chromosome': self.chromosome, 'fitness': self.fitness, 'probability': self.probability}


def fitnessFunc(chromosome=None):
    global TARGET_FIT

    clashes = 0
    clashes += abs(len(chromosome) - len(np.unique(chromosome)))

    for i in range(len(chromosome)):
        for j in range(len(chromosome)):
            if i != j:
                x = abs(i - j)
                y = abs(chromosome[i] - chromosome[j])
                if x == y:
                    clashes += 1

    return TARGET_FIT - clashes


def generateChromosome():
    global nQueens
    init_distribution = np.arange(nQueens)
    np.random.shuffle(init_distribution)
    return init_distribution


def generatePopulation(population_size):
    temp_pop = [createIndividual() for x in range(population_size)]
    for i in range(population_size):
        temp_pop[i].setChromosome(generateChromosome())
        temp_pop[i].setFitness(fitnessFunc(temp_pop[i].chromosome))
    return temp_pop


def generateProbability():
    global population
    sumFitness = np.sum([x.fitness for x in population])
    for x in population:
        x.setProbability(float(x.fitness / sumFitness))
        if debug_f:
            print(x.getAll())


def chooseRandom():
    global population
    global POPULATION
    p1 = None
    p2 = None

    generateProbability()

    while True:
        p1_prob = np.random.rand()
        if debug_f:
            print('p1_prob', p1_prob)
        sum = 0
        for x in population:
            sum += x.probability
            if sum > p1_prob:
                p1 = x
                break
        if p1 != None:
            break

    while True:
        p2_prob = np.random.rand()
        if debug_f:
            print('p2_prob', p2_prob)
        sum = 0
        for x in population:
            sum += x.probability
            if sum > p2_prob and p1 != x:
                p2 = x
                break
        if p2 != None:
            break

    if debug_f:
        print('p1', p1.getAll())
        print('p2', p2.getAll())
    return p1, p2


def mutate(individual):
    global MUTATION_RATE
    global POPULATION
    if np.random.rand() < MUTATION_RATE:
        a = np.random.randint(7)
        b = np.random.randint(8)
        individual.chromosome[a] = b
        individual.setFitness(fitnessFunc(individual.chromosome))

        if debug_m:
            print(a, b, individual.getAll())


def crossOver(par1, par2):
    cut_pos = np.random.randint(7)
    if debug_f:
        print('cut', cut_pos)
    child = createIndividual()
    child.chromosome = []
    child.chromosome.extend(par1.chromosome[0:cut_pos])
    child.chromosome.extend(par2.chromosome[cut_pos:])
    child.setFitness(fitnessFunc(child.chromosome))
    return child


def exitCond():
    global iter
    global population
    fitnessList = [x.fitness for x in population]
    if TARGET_FIT in fitnessList:
        print(population[fitnessList.index(TARGET_FIT)].getAll())
        return True
    elif MAX_ITER == iter:
        print('max over')
        return True
    return False


population = generatePopulation(POPULATION)

iter = 0
while not exitCond():
    if debug_f:
        print('while')
        for i in range(POPULATION):
            print('popu', population[i].getAll())
    new_population = []
    for i in range(POPULATION):
        x, y = chooseRandom()
        child = crossOver(x, y)
        mutate(child)
        if debug_f:
            print(i, 'x : ', x.getAll(), 'y : ', y.getAll())
            print(i, 'child : ', child.getAll() )
        new_population.append(child)
    iter += 1
    np.random.shuffle(new_population)
    if MUTATION_RATE > 1 / POPULATION:
        MUTATION_RATE = MUTATION_RATE / 2
    population = new_population

print(iter)
if debug_f:
    for i in range(POPULATION):
        print(i, new_population[i].getAll())
