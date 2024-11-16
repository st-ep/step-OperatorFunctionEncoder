import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
import matplotlib.colors as colors

def plot_displacement_magnitude(x, y, ux, uy, title, save_path=None, vmin=None, vmax=None, colorbar_label=None):
    """
    Plot displacement magnitude for a plate with circular hole.
    
    Parameters:
    -----------
    x : array-like
        x coordinates of the mesh points
    y : array-like
        y-component coordinates of the mesh points
    ux : array-like
        x-component of displacement
    uy : array-like
        y-component of displacement
    title : str
        Title for the plot
    save_path : str, optional
        If provided, saves the plot to this path
    vmin : float, optional
        Minimum value for colorbar scale
    vmax : float, optional
        Maximum value for colorbar scale
    colorbar_label : str, optional
        Label for the colorbar. If None, defaults to 'Displacement Magnitude'
    """
    # Create fine grid for interpolation
    xi = np.linspace(0, 1, 200)
    yi = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(xi, yi)
    
    # Calculate displacement magnitude
    displacement_magnitude = np.sqrt(ux**2 + uy**2)
    
    # Stack coordinates for interpolation
    points = np.column_stack((x, y))
    
    # Interpolate displacement magnitude onto regular grid
    method = 'linear'  # or 'nearest'
    U = griddata(points, displacement_magnitude, (X, Y), method=method, fill_value=np.nan)
    
    # Create mask for circular hole
    center = [0.5, 0.5]
    radius = 0.25
    mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2
    U = np.ma.masked_where(mask, U)
    
    # Define color normalization
    if vmin is not None and vmax is not None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Define contour levels explicitly based on vmin and vmax
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 51)  # 50 intervals
    else:
        levels = 50  # Default number of levels

    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot contours with defined normalization and levels
    if norm:
        contour = plt.contourf(X, Y, U, levels=levels, cmap='jet', norm=norm)
    else:
        contour = plt.contourf(X, Y, U, levels=levels, cmap='jet')
    
    # Create colorbar without altering its shape
    cbar = plt.colorbar(contour, label=colorbar_label or 'Displacement Magnitude')
    
    # Plot hole boundary
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius*np.cos(theta)
    circle_y = center[1] + radius*np.sin(theta)
    plt.plot(circle_x, circle_y, 'w-', linewidth=1.5)
    
    # Set plot properties
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def load_and_plot_predictions(json_file, groundtruth_save_path=None, b2b_save_path=None, deeponet_save_path=None, b2b_error_save_path=None, deeponet_error_save_path=None):
    """
    Load predictions and groundtruth from JSON file and create separate displacement plots
    
    Parameters:
    -----------
    json_file : str
        Path to the JSON file containing predictions
    groundtruth_save_path : str, optional
        If provided, saves the groundtruth plot to this path
    b2b_save_path : str, optional
        If provided, saves the B2B plot to this path
    deeponet_save_path : str, optional
        If provided, saves the DeepONet plot to this path
    b2b_error_save_path : str, optional
        If provided, saves the B2B error plot to this path
    deeponet_error_save_path : str, optional
        If provided, saves the DeepONet error plot to this path
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract coordinates and predictions
    xs = np.array(data['input_data']['xs'])
    ys = np.array(data['input_data']['ys'])  # groundtruth
    b2b_preds = np.array(data['b2b_predictions'])
    deeponet_preds = np.array(data['deeponet_predictions'])
    
    # Get coordinates for the first sample
    x_coords = xs[0, :, 0]  # x coordinates
    y_coords = xs[0, :, 1]  # y coordinates
    
    # Get displacements for the first sample
    groundtruth_ux = ys[0, :, 0]  # groundtruth x displacement
    groundtruth_uy = ys[0, :, 1]  # groundtruth y displacement
    b2b_ux = b2b_preds[0, :, 0]
    b2b_uy = b2b_preds[0, :, 1]
    deeponet_ux = deeponet_preds[0, :, 0]
    deeponet_uy = deeponet_preds[0, :, 1]
    
    # Plot groundtruth
    plot_displacement_magnitude(
        x_coords, y_coords, groundtruth_ux, groundtruth_uy,
        'Groundtruth Displacement Magnitude |u|',
        groundtruth_save_path
    )
    
    # Plot B2B predictions
    plot_displacement_magnitude(
        x_coords, y_coords, b2b_ux, b2b_uy,
        'B2B Displacement Magnitude |u|',
        b2b_save_path
    )
    
    # Plot DeepONet predictions
    plot_displacement_magnitude(
        x_coords, y_coords, deeponet_ux, deeponet_uy,
        'DeepONet Displacement Magnitude |u|',
        deeponet_save_path
    )
    
    # Calculate errors
    b2b_error_ux = b2b_ux - groundtruth_ux
    b2b_error_uy = b2b_uy - groundtruth_uy
    deeponet_error_ux = deeponet_ux - groundtruth_ux
    deeponet_error_uy = deeponet_uy - groundtruth_uy

    # Calculate common scale for error plots
    error_magnitudes = [
        np.sqrt(b2b_error_ux**2 + b2b_error_uy**2),
        np.sqrt(deeponet_error_ux**2 + deeponet_error_uy**2)
    ]
    vmin = 0
    vmax = max([np.max(mag) for mag in error_magnitudes])
    error_colorbar_label = 'Error Magnitude |u - u_true|'

    # Print max errors for debugging
    print(f"Max B2B error: {np.max(error_magnitudes[0])}")
    print(f"Max DeepONet error: {np.max(error_magnitudes[1])}")
    print(f"Using vmax: {vmax}")

    # Plot B2B error
    plot_displacement_magnitude(
        x_coords, y_coords, b2b_error_ux, b2b_error_uy,
        'B2B Error Magnitude |u_b2b - u_true|',
        b2b_error_save_path,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=error_colorbar_label
    )
    
    # Plot DeepONet error
    plot_displacement_magnitude(
        x_coords, y_coords, deeponet_error_ux, deeponet_error_uy,
        'DeepONet Error Magnitude |u_deeponet - u_true|',
        deeponet_error_save_path,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=error_colorbar_label
    )
    
    # Calculate and print maximum differences
    groundtruth_mag = np.sqrt(groundtruth_ux**2 + groundtruth_uy**2)
    b2b_mag = np.sqrt(b2b_ux**2 + b2b_uy**2)
    deeponet_mag = np.sqrt(deeponet_ux**2 + deeponet_uy**2)
    
    b2b_error = np.max(np.abs(b2b_mag - groundtruth_mag))
    deeponet_error = np.max(np.abs(deeponet_mag - groundtruth_mag))
    model_diff = np.max(np.abs(b2b_mag - deeponet_mag))
    
    print("\nMaximum differences in displacement magnitude:")
    print(f"B2B vs Groundtruth: {b2b_error:.6f}")
    print(f"DeepONet vs Groundtruth: {deeponet_error:.6f}")
    print(f"B2B vs DeepONet: {model_diff:.6f}")

# Function to check for invalid values
def check_invalid_values(array, name):
    if not np.all(np.isfinite(array)):
        print(f"Warning: {name} contains NaN or infinite values.")

# Example usage:
if __name__ == "__main__":
    json_file = "predictions_output/predictions_seed_1.json"
    load_and_plot_predictions(json_file)
