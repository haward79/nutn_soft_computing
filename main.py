
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def create_gif(filename: str, frames: np.ndarray, interval: int = 300, loop: int = 0) -> None:

    if len(filename) > 0:
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(filename, format="GIF", append_images=frames[1:], save_all=True, duration=interval, loop=loop)


def evaluate_problem(input_set: np.ndarray) -> np.ndarray:

    answer_set = []

    for (x, y, z) in input_set:
        answer = 20 + np.exp(1) - 20 * np.exp(-0.2 * (1/3 * (x**2 + y**2 + z**2))**0.5) - np.exp(1/3 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)))
        answer_set.append(answer)

    return np.array(answer_set)


def pso_algorithm(pop_size: int, w: float, c1: float, c2: float, left_boundary: float, right_boundary: float, max_iteration: int = 30) -> tuple:

    # Init particles randomly.
    position = np.random.uniform(left_boundary, right_boundary, (pop_size, 3))
    vector = np.random.uniform(0, 1, (pop_size, 3))

    pbest_position = np.zeros((pop_size, 3))
    pbest_fitness = np.full((pop_size,), float("inf"), float)
    gbest_position = np.zeros((3,))
    gbest_fitness = None
    results = []
    frames = []

    for gen_id in range(max_iteration):
        # Move particles according to vector.
        position += vector

        # Evaluate fitness.
        fitness = evaluate_problem(position)

        # Plot particles for this generation.
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(position[:, 0], position[:, 1], position[:, 2])

        for i, label in enumerate(fitness):
            ax.text(position[i][0], position[i][1], position[i][2], '%.2f' % (label,))

        ax.set_xlim(left_boundary, right_boundary)
        ax.set_ylim(left_boundary, right_boundary)
        ax.set_zlim(left_boundary, right_boundary)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
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

        # Save result of this generation.
        results.append([gbest_position, gbest_fitness])

        # Update vector for next generation.
        vector = w * vector + c1 * np.random.uniform(0, 1) * (pbest_position-position) + c2 * np.random.uniform(0, 1) * (gbest_position-position)

    return (gbest_position, gbest_fitness, results, frames)


def pso_demo():

    # Run pso algorithm.
    (gbest_position, gbest_fitness, results, frames) = pso_algorithm(pop_size=10, w=0.8, c1=0.8, c2=0.2, left_boundary=-32, right_boundary=32, max_iteration=30)

    # Print table of records.
    print('+-----+--------+--------------------------+')
    for (index, (point, fitness)) in enumerate(results):
        print('| %3d   %6.2f   (%6.2f, %6.2f, %6.2f) |' % (index+1, fitness, point[0], point[1], point[2]))
    print('+-----+--------+--------------------------+')

    print()
    print('There is a min, %.2f, at (%.2f, %.2f, %.2f) !' % (gbest_fitness, gbest_position[0], gbest_position[1], gbest_position[2]))

    # Create animation of records.
    create_gif('pso.gif', frames)


pso_demo()
