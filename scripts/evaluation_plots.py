import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_csv_lines(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Check if the data has at least 6 columns
    if data.shape[1] < 5:
        raise ValueError("CSV file must contain at least 5 columns.")
    
    # Extract the x values and the y values
    y_columns = data.columns[0:]  # Get column names for y values
    
    # Create subplots
    fig, axes = plt.subplots(len(y_columns), 1, figsize=(10, 12), sharex=True)
    
    x = np.arange(0, 501, 25)

    # Plot each y column against x
    for i, col in enumerate(y_columns):
        axes[i].plot(x, data[col], label=col)
        axes[i].set_ylabel(col)
        axes[i].legend(loc='upper right')
    
    # Set the x label only on the bottom plot
    axes[-1].set_xlabel('epoch')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('/export/livia/home/vision/Ymohammadi/Code/results/evaluation.png')

# Example usage
csv_file = '/export/livia/home/vision/Ymohammadi/Code/results/evaluation.csv'
plot_csv_lines(csv_file)