# Utility for plotting weighting functions in TP system
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import matplotlib.pyplot as plt

__all__ = [
    'draw_weighting_system'
]

def draw_weighting_system(Us, coord_grid):
    pairs = [(coord_grid.get_coords_per_dim()[uinx], Us[uinx]) for uinx in range(len(Us))]

    # Iterate through the pairs and create separate figures
    for i, (x, y) in enumerate(pairs, start=1):
        plt.figure(i)  # Create a new figure

        num_polylines = len(y[0])
        plt.plot(x, y, marker='o', label=[f"Subtensor in dim {x}" for x in range(1, num_polylines+1)])  # Plot the data
        plt.title(f'{coord_grid.get_dim_names()[i-1]}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()  # Show the plot in a new window
