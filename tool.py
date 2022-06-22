
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


_tee_msg = ''


def tee(msg: str = '') -> None:

    global _tee_msg

    _tee_msg += msg + '\n'
    print(msg)


def write_tee(filename: str) -> None:

    global _tee_msg

    with open(filename + '.txt', 'w') as fout:
        fout.write(_tee_msg)

    _tee_msg = ''


def evaluate_problem(input_set: np.ndarray) -> np.ndarray:

    answer_set = []

    for (x, y, z) in input_set:
        answer = 20 + np.exp(1) - 20 * np.exp(-0.2 * (1/3 * (x**2 + y**2 + z**2))**0.5) - np.exp(1/3 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)))
        answer_set.append(answer)

    return np.array(answer_set)


def plot_population(generation_id: int, population: np.ndarray, fitness: np.ndarray, left_boundary: float, right_boundary: float) -> np.ndarray:

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(population[:, 0], population[:, 1], population[:, 2])
    ax.set_title('Generation ' + str(generation_id + 1))

    for i, label in enumerate(fitness):
        pass
        #ax.text(population[i][0], population[i][1], population[i][2], '%.2f' % (label,))

    ax.set_xlim(left_boundary, right_boundary)
    ax.set_ylim(left_boundary, right_boundary)
    ax.set_zlim(left_boundary, right_boundary)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return frame


def create_gif(filename: str, frames: np.ndarray, interval: int = 300, loop: int = 0) -> None:

    if len(filename) > 0:
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(filename+'.gif', format="GIF", append_images=frames[1:], save_all=True, duration=interval, loop=loop)


def print_result(gbest_position: np.ndarray, gbest_fitness: float, results: list) -> None:

    tee('+-----+----------+-----------------------------------+')
    for (index, (point, fitness)) in enumerate(results):
        tee('| %3d   %8.4f   (%9.4f, %9.4f, %9.4f) |' % (index + 1, fitness, point[0], point[1], point[2]))
    tee('+-----+----------+-----------------------------------+')

    tee()
    tee('There is a min, %.4f, at (%.4f, %.4f, %.4f) !' % (gbest_fitness, gbest_position[0], gbest_position[1], gbest_position[2]))
