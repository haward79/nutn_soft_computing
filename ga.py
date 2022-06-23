
import numpy as np
from tool import *


FLOAT_BITS = 6
NUM_BITS = 28
RIGHT_BOUNDARY = (2 ** (NUM_BITS - 1) - 1) * 0.5
LEFT_BOUNDARY = -RIGHT_BOUNDARY


def bin2dec(bin: str) -> float:

    dec = int(bin[1:], 2)

    if bin.startswith('1'):
        dec = -dec

    dec = dec / 10 ** FLOAT_BITS

    return dec


def dec2bin(dec: float) -> str:

    sign = '0'

    if dec < 0:
        dec = -dec
        sign = '1'

    dec *= 10 ** FLOAT_BITS
    dec = int(dec)

    bin = ('{:0' + str(NUM_BITS - 1) + 'b}').format(dec)
    assert len(bin) == NUM_BITS - 1

    bin = sign + bin

    return bin


def pheno2geno(phenos: np.ndarray) -> np.ndarray:

    genos = []

    for pheno in phenos:
        assert len(pheno) == 3

        geno = ''
        for i in range(3):
            geno += dec2bin(pheno[i])

        genos.append(geno)

    return np.array(genos)


def geno2pheno(genos: np.ndarray) -> np.ndarray:

    phenos = []

    for geno in genos:
        assert len(geno) % 3 == 0
        size = len(geno) // 3

        pheno = [bin2dec(geno[:size]), bin2dec(geno[size:size*2]), bin2dec(geno[size*2:])]

        phenos.append(pheno)

    return np.array(phenos)


def choose_parents(population: np.ndarray) -> np.ndarray:

    fitness = evaluate_problem(population)
    probability = fitness / fitness.sum()

    indices = np.random.choice(np.arange(len(population)), (len(population),), p=probability)
    parents = population[indices].copy()

    return parents


def crossover(parents: np.ndarray, prob: float) -> np.ndarray:

    do_crossover_prob = prob / 2
    children = []

    for i in range(0, len(parents), 2):
        # Parents.
        p1 = parents[i]
        p2 = parents[i + 1]

        # Do crossover.
        if np.random.choice([False, True], p=[1-do_crossover_prob, do_crossover_prob]):
            # Choose a break point.
            assert len(p1) >= 2
            index = np.random.choice(np.arange(len(p1) - 1)) + 1

            # Decide whether to do a swap.
            if np.random.choice([False, True], p=[1-prob, prob]):
                n1 = p2[index:] + p1[:index]
                n2 = p1[index:] + p2[:index]
                children.append(n1)
                children.append(n2)

            else:
                n1 = p1[:index] + p2[index:]
                n2 = p2[:index] + p1[index:]
                children.append(n1)
                children.append(n2)

        # Don't do crossover.
        else:
            # Only save one of the parent.
            if np.random.uniform(0, 1) >= 0.5:
                children.append(p1)
                children.append(p1)
            else:
                children.append(p2)
                children.append(p2)

    return np.array(children)


def mutate(children: np.ndarray, prob: float) -> np.ndarray:

    FLIP = {
        '0': '1',
        '1': '0'
    }

    mutated = []

    for child in children:
        tmp = ''

        for bit in child:
            if np.random.choice([False, True], p=[1-prob, prob]):
                tmp += FLIP[bit]
            else:
                tmp += bit

        mutated.append(tmp)

    return mutated


def survival_select(parents: np.ndarray, children: np.ndarray, num: int) -> np.ndarray:

    num_parent = int(num * 0.8)
    num_children = num - num_parent

    parent_selected = parents[np.argsort(evaluate_problem(parents))[:num_parent]].copy()
    children_selected = children[np.argsort(evaluate_problem(children))[:num_children]].copy()

    return np.concatenate((parent_selected, children_selected), axis=0)


def ga_algorithm(pop_size: int, crossover_prob: float, mutation_prob: float, left_boundary: float, right_boundary: float, max_iteration: int = 30, term_diff: float = 0.1) -> tuple:

    # Init population randomly and evaluate their fitness.
    population = np.random.uniform(left_boundary, right_boundary, (pop_size, 3))

    gen_solution = None
    gen_fitness = None
    results = []
    frames = []
    count_convergence = 0

    for gen_id in range(max_iteration):
        # Choose parents.
        parents = choose_parents(population)

        # Mate.
        children = crossover(pheno2geno(parents), crossover_prob)

        # Mutate.
        children = mutate(children, mutation_prob)

        for i in range(len(children)):
            child = children[i]
            children[i] = child

        # Survival selection.
        population = survival_select(parents, geno2pheno(children), len(parents))

        # Evaluate fitness.
        fitness = evaluate_problem(population)

        # Plot particles for this generation.
        frame = plot_population(gen_id, population, fitness, left_boundary*2, right_boundary*2)
        frames.append(frame)

        # Get the best solution in this generation.
        index = fitness.argmax()
        gen_solution = population[index]
        gen_fitness = fitness[index]

        # Check if the result is convergent in this generation.
        if len(results) > 0 and abs(gen_fitness - results[-1][1]) < term_diff:
            count_convergence += 1

        # Save result for this generation.
        results.append([gen_solution, gen_fitness])

        # Check if the result is convergent for 10 times continuously.
        if count_convergence >= 10:
            break

    results_np = np.array(results, dtype=object)
    index = results_np[:, 1].argmin()
    gbest_fitness = results_np[index, 1]
    gbest_position = results_np[index, 0].copy()

    return (gbest_position, gbest_fitness, results, frames)


def ga_demo():

    TIMES = 10
    LEFT_BOUNDARY = -32
    RIGHT_BOUNDARY = 32
    solution = []
    fitness = []

    # Run ten times.
    for i in range(TIMES):
        # Run pso algorithm.
        (gbest_position, gbest_fitness, results, frames) = ga_algorithm(pop_size=50, crossover_prob=0.6, mutation_prob=0.1, left_boundary=LEFT_BOUNDARY, right_boundary=RIGHT_BOUNDARY, max_iteration=100, term_diff=0.1)
        solution.append(gbest_position)
        fitness.append(gbest_fitness)

        # Print table of records.
        print_result(gbest_position, gbest_fitness, results)
        tee('\n')

        # Plot fitness for each generation.
        plot_fitness(i + 1, np.array(results, dtype=object)[:, 1], LEFT_BOUNDARY, RIGHT_BOUNDARY, 'output/ga_fitness' + str(i + 1))

        # Create animation of records.
        create_gif('output/ga_population' + str(i + 1), frames)

    # Write tee to disk.
    write_tee('output/ga_gens')

    # Convert list to numpy array for calculation convenience.
    solution = np.array(solution)
    fitness = np.array(fitness)

    # Calculate mean and standard deviation.
    mean = fitness.sum() / TIMES
    std_deviation = ((fitness - mean) ** 2).sum() / TIMES

    tee('Mean = %.4f' % (mean,))
    tee('Standard Deviation = %.4f' % (std_deviation,))
    write_tee('output/ga_summary')


ga_demo()
