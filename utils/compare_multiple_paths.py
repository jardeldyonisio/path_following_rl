#!/usr/bin/env python3
#coding: utf-8

import os
import matplotlib.pyplot as plt
import numpy as np

'''

A

Original path: /home/lognav/Jardel/path_following_rl/paths/path_circular.txt

A - T&R:

1: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-34-42/following_data.txt
2: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-39-43/following_data.txt
3: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-42-04/following_data.txt
Bezier: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-42-04/bezier_path_coords_data.txt

A - RL

1: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_circular_2025-12-07_10-34-57.txt
2: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_circular_2025-12-07_10-35-00.txt
3: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_circular_2025-12-07_10-35-03.txt

------
B

Original path: /home/lognav/Jardel/path_following_rl/paths/path_eight.txt

B - T&R:
1: /home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-31-49/following_data.txt
2: /home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-36-14/following_data.txt
3: /home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-41-34/following_data.txt
Bezier: /home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-41-34/bezier_path_coords_data.txt

B - RL
1: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_eight_2025-12-07_10-41-31.txt
2: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_eight_2025-12-07_10-41-39.txt
3: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_eight_2025-12-07_10-41-46.txt

------
C

Original path: /home/lognav/Jardel/path_following_rl/paths/path_corredor.txt

C - T&R:

1: /home/lognav/Jardel/shark-mb-ros/data/16-04-2024_15-41-31/following_data.txt
2: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_12-34-15/following_data.txt
3: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_12-52-19/following_data.txt
Bezier: /home/lognav/Jardel/shark-mb-ros/data/18-04-2024_12-52-19/bezier_path_coords_data.txt

C - RL

1: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_corredor_2025-12-07_10-44-13.txt
2: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_corredor_2025-12-07_10-44-20.txt
3: /home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_corredor_2025-12-07_10-44-27.txt

'''

def carregar_dados_do_arquivo(file_path):
    dados = np.loadtxt(file_path, delimiter=',')
    x = dados[:, 0]
    y = dados[:, 1]
    return x, y

def compare_multiple_paths(original_path, following_ter_data_1, following_ter_data_2, following_ter_data_3, 
                           following_rl_data_1, following_rl_data_2, following_rl_data_3, bezier_path_coords_data):
    # Obter diretório de saída
    output_base = os.path.join("/home/lognav/Jardel/path_following_rl/results", "path_comparison_corredor")

    # Carregar dados
    x1, y1 = carregar_dados_do_arquivo(original_path)
    x2, y2 = carregar_dados_do_arquivo(following_ter_data_1)
    x3, y3 = carregar_dados_do_arquivo(following_ter_data_2)
    x4, y4 = carregar_dados_do_arquivo(following_ter_data_3)
    x5, y5 = carregar_dados_do_arquivo(following_rl_data_1)
    x6, y6 = carregar_dados_do_arquivo(following_rl_data_2)
    x7, y7 = carregar_dados_do_arquivo(following_rl_data_3)
    x8, y8 = carregar_dados_do_arquivo(bezier_path_coords_data)

    # Plotar gráfico
    plt.figure(figsize=(10, 10))
    plt.plot(x1, y1, label='Original Path')
    plt.plot(x2, y2, label='Path Following T&R 1', linestyle='--', color='b')
    plt.plot(x3, y3, label='Path Following T&R 2', linestyle='--', color='g')
    plt.plot(x4, y4, label='Path Following T&R 3', linestyle='--', color='m')
    plt.plot(x5, y5, label='Path Following RL 1', linestyle=':', color='c')
    plt.plot(x6, y6, label='Path Following RL 2', linestyle=':', color='y')
    plt.plot(x7, y7, label='Path Following RL 3', linestyle=':', color='k')
    plt.plot(x8, y8, label='Bézier Path', linestyle='-.', color='r')

    plt.title('Corridor Path Comparison')
    plt.xlabel('Axis X (m)')
    plt.ylabel('Axis Y (m)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 30)
    plt.ylim(-5, 5)
    plt.axis('equal')  # Garante proporção quadrada
    plt.savefig(output_base + '.png')
    plt.savefig(output_base + '.pdf')
    plt.show()

if __name__ == "__main__":
    # Circular path comparison
    # original_path = '/home/lognav/Jardel/path_following_rl/paths/path_circular.txt'
    # following_ter_data_1 = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-34-42/following_data.txt'
    # following_ter_data_2 = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-39-43/following_data.txt'
    # following_ter_data_3 = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-42-04/following_data.txt'
    # following_rl_data_1 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_circular_2025-12-07_10-34-57.txt'
    # following_rl_data_2 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_circular_2025-12-07_10-35-00.txt'
    # following_rl_data_3 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_circular_2025-12-07_10-35-03.txt'
    # bezier_path_coords_data = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_13-42-04/bezier_path_coords_data.txt'

    # Eight path comparison
    # original_path = '/home/lognav/Jardel/path_following_rl/paths/path_eight.txt'
    # following_ter_data_1 = '/home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-31-49/following_data.txt'
    # following_ter_data_2 = '/home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-36-14/following_data.txt'
    # following_ter_data_3 = '/home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-41-34/following_data.txt'
    # following_rl_data_1 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_eight_2025-12-07_10-41-31.txt'
    # following_rl_data_2 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_eight_2025-12-07_10-41-39.txt'
    # following_rl_data_3 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_eight_2025-12-07_10-41-46.txt'
    # bezier_path_coords_data = '/home/lognav/Jardel/shark-mb-ros/data/22-04-2024_15-41-34/bezier_path_coords_data.txt'

    # Corridor path comparison
    original_path = '/home/lognav/Jardel/path_following_rl/paths/path_corredor.txt'
    following_ter_data_1 = '/home/lognav/Jardel/shark-mb-ros/data/16-04-2024_15-41-31/following_data.txt'
    following_ter_data_2 = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_12-34-15/following_data.txt'
    following_ter_data_3 = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_12-52-19/following_data.txt'
    following_rl_data_1 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_corredor_2025-12-07_10-44-13.txt'
    following_rl_data_2 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_corredor_2025-12-07_10-44-20.txt'
    following_rl_data_3 = '/home/lognav/Jardel/path_following_rl/trajectories/trajectory_path_corredor_2025-12-07_10-44-27.txt'
    bezier_path_coords_data = '/home/lognav/Jardel/shark-mb-ros/data/18-04-2024_12-52-19/bezier_path_coords_data.txt'

    compare_multiple_paths(original_path, following_ter_data_1, following_ter_data_2, following_ter_data_3, following_rl_data_1, 
                           following_rl_data_2, following_rl_data_3, bezier_path_coords_data)