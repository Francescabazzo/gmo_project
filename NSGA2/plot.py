# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter


def plot_pareto_front_2d(pareto_front: np.ndarray,
                         objectives_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates 2D scatter plots showing all pairwise combinations of the three objectives
    from the pareto front.

    Args:
        pareto_front: NumPy array of shape (n_solutions, 3) containing objective values
        objectives_names: List of three objective names. Defaults to time differences,
                         assignment differences, and makespan
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if objectives_names is None:
        objectives_names = ['Time Differences', 'Assignment Differences', 'Makespan']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot objective pairs
    pairs = [(0, 1), (0, 2), (1, 2)]

    for i, (obj1, obj2) in enumerate(pairs):
        axes[i].scatter(pareto_front[:, obj1], pareto_front[:, obj2],
                        alpha=0.7, s=50, color='blue')
        axes[i].set_xlabel(objectives_names[obj1])
        axes[i].set_ylabel(objectives_names[obj2])
        axes[i].set_title(f'{objectives_names[obj1]} vs {objectives_names[obj2]}')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto front 2D saved to: {save_path}")

    return fig


def plot_pareto_front_3d_matplotlib(pareto_front: np.ndarray,
                                    objectives_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None,
                                    show_labels: bool = True,
                                    assess: bool = False,
                                    optimal_pareto_front: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a 3D visualization of the Pareto front with color coding based on the number
    of solutions at each coordinate point.

    Args:
        pareto_front: NumPy array of shape (n_solutions, 3) containing objective values
        objectives_names: List of three objective names. Defaults to time differences,
                         assignment differences, and makespan
        save_path: Optional path to save the figure
        show_labels: Whether to show point labels (currently disabled for clarity)

    Returns:
        Tuple containing matplotlib figure and axes objects
    """
    if objectives_names is None:
        objectives_names = ['Time Differences', 'Assignment Differences', 'Makespan']

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Count how many solutions share the same coordinates
    point_counts = Counter([tuple(point) for point in pareto_front])

    # Create color array based on count of solutions at each point
    colors = [point_counts[tuple(point)] for point in pareto_front]

    # Plot points with color based on number of solutions at that point
    scatter1 = ax.scatter(
        pareto_front[:, 0],
        pareto_front[:, 1],
        pareto_front[:, 2],
        c=colors,  # Color by number of solutions at this point
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    if assess == True and optimal_pareto_front is not None:
        scatter2 = ax.scatter(
            optimal_pareto_front[:, 0],
            optimal_pareto_front[:, 1],
            optimal_pareto_front[:, 2],
            c='red',
            s=100,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )

    # Labels
    ax.set_xlabel(objectives_names[0], fontsize=12, labelpad=10)
    ax.set_ylabel(objectives_names[1], fontsize=12, labelpad=10)
    ax.set_zlabel(objectives_names[2], fontsize=12, labelpad=10)

    ax.set_title(f'Pareto Front 3D - {len(pareto_front)} Solutions', fontsize=14, pad=20)

    # Colorbar
    cbar = plt.colorbar(scatter1, ax=ax, pad=0.1)
    cbar.set_label('Number of Solutions', rotation=270, labelpad=15)

    # Grid
    ax.grid(True, alpha=0.3)

    # Improve viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Pareto front 3D (matplotlib) saved to: {save_path}")

    # plt.show()
    return fig, ax