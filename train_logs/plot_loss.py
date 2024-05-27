import re
import os
import matplotlib.pyplot as plt

def plot_loss_from_log(log_file):
    with open(log_file, 'r') as file:
        log = file.read()

    iterations = re.findall(r'iter (\d+)', log)
    losses = re.findall(r'loss: (\d+\.\d+)', log)

    iterations = [int(iteration) for iteration in iterations]
    losses = [float(loss) for loss in losses]

    return iterations, losses

def plot_losses(log_files, output_dir):
    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        iterations, losses = plot_loss_from_log(log_file)
        plt.plot(iterations, losses, label=os.path.splitext(os.path.basename(log_file))[0])

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified output directory
    output_file = os.path.join(output_dir, 'training_loss_plot.png')
    plt.savefig(output_file)
    print(f"Plot saved as '{output_file}'")

    plt.show()

# Specify the paths to your log files
log_files = [
    # Quantitative Results
    'pos_martial_training_log_MSE_loss.txt',
    'euler_martial_training_log.txt',
    '6D_martial_training_log.txt',
    'quad_martial_training_log.txt'

    # Qualitative Results for Research questions
    # 'pos_martial_training_log_MSE_loss.txt',
    # 'pos_martial_training_log_L1_loss.txt',
    # 'pos_martial_training_log_angle_distance_loss.txt'
]

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Call the plot_losses function with the log files and output directory
plot_losses(log_files, script_dir)