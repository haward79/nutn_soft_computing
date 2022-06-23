
import numpy as np
from tool import *


def pso_algorithm(pop_size: int, w: float, c1: float, c2: float, left_boundary: float, right_boundary: float, max_iteration: int = 30, term_diff: float = 0.1) -> tuple:

    # Init particles randomly.
    position = np.random.uniform(left_boundary, right_boundary, (pop_size, 3))
    vector = np.random.uniform(0, 1, (pop_size, 3))

    pbest_position = np.zeros((pop_size, 3))
    pbest_fitness = np.full((pop_size,), float("inf"), float)
    gbest_position = np.zeros((3,))
    gbest_fitness = None
    results = []
    frames = []
    count_convergence = 0

    for gen_id in range(max_iteration):
        # Move particles according to vector.
        position += vector

        # Evaluate fitness.
        fitness = evaluate_problem(position)

        # Plot particles for this generation.
        frame = plot_population(gen_id, position, fitness, left_boundary, right_boundary)
        frames.append(frame)

        # Update pbest.
        for (index, update_required) in enumerate(fitness < pbest_fitness):
            if update_required:
                pbest_position[index] = np.copy(position[index])
                pbest_fitness[index] = fitness[index]

        # Update gbest.
        index = pbest_fitness.argmin()
        gbest_position = np.copy(pbest_position[index])
        gbest_fitness = pbest_fitness[index]

        # Check if the result is convergent in this generation.
        if len(results) > 0 and abs(gbest_fitness - results[-1][1]) < term_diff:
            count_convergence += 1

        # Save result for this generation.
        results.append([gbest_position, gbest_fitness])

        # Check if the result is convergent for 10 times continuously.
        if count_convergence >= 10:
            break

        # Update vector for next generation.
        vector = w * vector + c1 * np.random.uniform(0, 1) * (pbest_position-position) + c2 * np.random.uniform(0, 1) * (gbest_position-position)

    return (gbest_position, gbest_fitness, results, frames)


def pso_demo():

    TIMES = 10
    LEFT_BOUNDARY = -32
    RIGHT_BOUNDARY = 32
    solution = []
    fitness = []

    # Run ten times.
    for i in range(TIMES):
        # Run pso algorithm.
        (gbest_position, gbest_fitness, results, frames) = pso_algorithm(pop_size=10, w=0.8, c1=0.8, c2=0.2, left_boundary=LEFT_BOUNDARY, right_boundary=RIGHT_BOUNDARY, max_iteration=100, term_diff=0.001)
        solution.append(gbest_position)
        fitness.append(gbest_fitness)

        # Print table of records.
        print_result(gbest_position, gbest_fitness, results)
        tee('\n')

        # Plot fitness for each generation.
        plot_fitness(i+1, np.array(results, dtype=object)[:, 1], LEFT_BOUNDARY, RIGHT_BOUNDARY, 'output/pso_fitness' + str(i+1))

        # Create animation of records.
        create_gif('output/pso_population' + str(i+1), frames)

    # Write tee to disk.
    write_tee('output/pso_gens')

    # Convert list to numpy array for calculation convenience.
    solution = np.array(solution)
    fitness = np.array(fitness)

    # Calculate mean and standard deviation.
    mean = fitness.sum() / TIMES
    std_deviation = ((fitness - mean) ** 2).sum() / TIMES

    tee('Mean = %.4f' % (mean,))
    tee('Standard Deviation = %.4f' % (std_deviation,))
    write_tee('output/pso_summary')


pso_demo()
