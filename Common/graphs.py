import matplotlib.pyplot as plt
import numpy as np

plt.ion()


def show_loss_curves(training_losses, validation_losses, epochs, running_average_training_losses, running_average_validation_losses, running_average_epochs, save_path):

    # Create the figure
    fig = plt.figure(num=0, tight_layout=True)
    plt.clf()
    fig.canvas.set_window_title('Losses')
    plt.title('Training and Validation Losses')
    plt.gca().set_yscale('log')

    # Plot the training losses
    plt.plot(epochs, training_losses, label='Training', color=(0.6, 0.6, 0.8))
    if len(running_average_epochs) > 0:
        minimum_loss = round(min(running_average_training_losses), 4)
        final_loss = round(running_average_training_losses[-1], 4)
        plt.plot(running_average_epochs, running_average_training_losses, label='Training Run Ave: Min = ' + str(minimum_loss) + ', Final = ' + str(final_loss), color=(0, 0, 0.5))

    # Plot the validation losses
    if len(validation_losses) > 0:
        plt.plot(epochs, validation_losses, label='Validation', color=(0.6, 0.8, 0.6))
        if len(running_average_epochs) > 0:
            minimum_loss = round(min(running_average_validation_losses), 4)
            final_loss = round(running_average_validation_losses[-1], 4)
            plt.plot(running_average_epochs, running_average_validation_losses, label='Validation Run Ave: Min = ' + str(minimum_loss) + ', Final = ' + str(final_loss), color=(0, 0.5, 0))

    # Add a legend
    plt.legend()

    # Save and show
    plt.savefig(save_path)
    plt.show()
    plt.pause(0.01)


def show_error_curves(validation_errors, epochs, save_path):

    validation_errors = np.array(validation_errors)
    validation_x_errors = validation_errors[:, 0]
    validation_y_errors = validation_errors[:, 1]
    validation_theta_errors = validation_errors[:, 2]

    fig = plt.figure(1)
    plt.clf()
    fig.canvas.set_window_title('Validation Errors')
    plt.title('X, Y, and Theta Validation Errors')
    ax1 = plt.gca()
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Position')
    ax1.plot(epochs, validation_x_errors, label='X Error (current = ' + str(np.round(1000 * validation_x_errors[-1], 1)) + ' mm)', color=(0.8, 0.5, 0.5))
    ax1.plot(epochs, validation_y_errors, label='Y Error (current = ' + str(np.round(1000 * validation_y_errors[-1], 1)) + ' mm)', color=(0.5, 0.8, 0.5))
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('Orientation')
    ax2.plot(epochs, validation_theta_errors, label='Theta Error (current = ' + str(np.round(np.rad2deg(validation_theta_errors[-1]), 1)) + ' deg)', color=(0.5, 0.5, 0.8))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path)
    plt.pause(0.01)


def show_velocity_error_curves(validation_errors, epochs, save_path):

    validation_errors = np.array(validation_errors)
    validation_translation_errors = validation_errors[:, 0]
    validation_rotation_errors = validation_errors[:, 1]

    fig = plt.figure(2)
    plt.clf()
    fig.canvas.set_window_title('Validation Errors')
    plt.title('Translation and Rotation Velocity Validation Errors')
    ax1 = plt.gca()
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Position')
    ax1.plot(epochs, validation_translation_errors, label='Position Error (current = ' + str(np.round(1000 * validation_translation_errors[-1], 1)) + ' mm)', color=(0.8, 0.5, 0.5))
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('Rotation')
    ax2.plot(epochs, validation_rotation_errors, label='Rotation Error (current = ' + str(np.round(np.rad2deg(validation_rotation_errors[-1]), 1)) + ' deg)', color=(0.5, 0.8, 0.5))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path)
    plt.pause(0.01)


def show_xy_errors(xy_errors):

    fig = plt.figure(num=3)
    plt.clf()
    fig.canvas.set_window_title('XY Errors')
    plt.title('XY Errors')
    plt.imshow(xy_errors)
    plt.colorbar()
    plt.pause(0.01)


def show_uncertainties_vs_poses(uncertainties, poses, graph_path):

    # Uncertainties are: [x, y, theta]
    # Poses are: [x, y, theta, z]

    fig, (ax_x, ax_y, ax_z, ax_theta) = plt.subplots(nrows=4, ncols=1, num=4, figsize=(5, 10), tight_layout=True)
    fig.canvas.set_window_title('Uncertainties vs Poses')

    ax_x.set_title('Uncertainty vs X')
    ax_x.set_xlabel('X')
    ax_x.set_ylabel('X / Y Uncertainty')
    ax_x.scatter(poses[:, 0], uncertainties[:, 0], label='X Uncertainty')
    ax_x.scatter(poses[:, 0], uncertainties[:, 1], label='Y Uncertainty')
    ax_x_2 = ax_x.twinx()
    ax_x_2.set_ylabel('Theta Uncertainty')
    ax_x_2.scatter(poses[:, 0], uncertainties[:, 2], label='Theta Uncertainty')

    ax_y.set_title('Uncertainties vs Y')
    ax_y.set_xlabel('Y')
    ax_y.set_ylabel('X / Y Uncertainty')
    ax_y.scatter(poses[:, 1], uncertainties[:, 0], label='X Uncertainty')
    ax_y.scatter(poses[:, 1], uncertainties[:, 1], label='Y Uncertainty')
    ax_y_2 = ax_y.twinx()
    ax_y_2.set_ylabel('Theta Uncertainty')
    ax_y_2.scatter(poses[:, 1], uncertainties[:, 2], label='Theta Uncertainty')

    ax_z.set_title('Uncertainties vs Z')
    ax_z.set_xlabel('Z')
    ax_z.set_ylabel('X / Y Uncertainty')
    ax_z.scatter(poses[:, 3], uncertainties[:, 0], label='X Uncertainty')
    ax_z.scatter(poses[:, 3], uncertainties[:, 1], label='Y Uncertainty')
    ax_z_2 = ax_z.twinx()
    ax_z_2.set_ylabel('Theta Uncertainty')
    ax_z_2.scatter(poses[:, 3], uncertainties[:, 2], label='Theta Uncertainty')

    ax_theta.set_title('Uncertainties vs Theta')
    ax_theta.set_xlabel('Theta')
    ax_theta.set_ylabel('X / Y Uncertainty')
    ax_theta.scatter(poses[:, 2], uncertainties[:, 0], label='X Uncertainty')
    ax_theta.scatter(poses[:, 2], uncertainties[:, 1], label='Y Uncertainty')
    ax_theta_2 = ax_theta.twinx()
    ax_theta_2.set_ylabel('Theta Uncertainty')
    ax_theta_2.scatter(poses[:, 2], uncertainties[:, 2], label='Theta Uncertainty')

    plt.show()
    plt.savefig(graph_path)
    plt.pause(100)
