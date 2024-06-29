import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# Define initial values for the transformation matrix
initial_values = {'m11': 1, 'm12': 0, 'm21': 0, 'm22': 1}
previous_eigen = []
colors = ['blue', 'green']

# Generate a grid in the 2D plane
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()])

# Setup the figure and axes for plotting
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.25, bottom=0.35)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_title('2D Linear Transformation')
legend = ax.legend()

# Draw the initial grid lines
lines = []
for i in range(len(x)):
    line, = ax.plot(X[i, :], Y[i, :], 'gray', alpha=0.5)
    lines.append(line)
    line, = ax.plot(X[:, i], Y[:, i], 'gray', alpha=0.5)
    lines.append(line)

transformed_lines = []
for _ in range(len(lines)):
    line, = ax.plot([], [], 'r-', alpha=0.5)
    transformed_lines.append(line)

# Text for displaying the determinant
det_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)


# Function to apply transformation matrix to points
def apply_transformation(matrix, points):
    return np.dot(matrix, points)


def plot_eigen(matrix, show):
    global previous_eigen
    global legend
    # Clear previous arrows
    for eigen in previous_eigen:
        eigen.remove()
    previous_eigen = []

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Plot the eigenvectors
    if show:
        for i in range(len(eigenvalues)):
            start_point = [0, 0]
            end_point = eigenvectors[:, i] * eigenvalues[i]
            eigen = ax.quiver(start_point[0], start_point[1], end_point[0], end_point[1], scale=1, scale_units='xy',
                              angles='xy',
                              color=colors[i],
                              label=f'Eigenvector {i + 1} with $\lambda={eigenvalues[i]:.3f}$')
            previous_eigen.append(eigen)
        ax.legend()
    else:
        legend.remove()


# Update function for the plot and determinant
def update(val):
    show = show_eigen.get_status()[0]  # Get the status of the checkbox
    m11, m12, m21, m22 = sm11.val, sm12.val, sm21.val, sm22.val
    transform_matrix = np.array([[m11, m12], [m21, m22]])
    det = np.linalg.det(transform_matrix)
    det_text.set_text(f'Determinant: {det:.2f}')
    transformed_points = apply_transformation(transform_matrix, points).reshape(2, len(x), len(y))
    plot_eigen(transform_matrix, show)

    for i in range(len(x)):
        transformed_lines[2 * i].set_data(transformed_points[0, i, :], transformed_points[1, i, :])
        transformed_lines[2 * i + 1].set_data(transformed_points[0, :, i], transformed_points[1, :, i])

    fig.canvas.draw_idle()


# Reset function to restore initial values and update determinant
def reset(event):
    sm11.reset()
    sm12.reset()
    sm21.reset()
    sm22.reset()


# Create sliders for transformation matrix elements
axcolor = 'lightgoldenrodyellow'
ax_m11 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_m12 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_m21 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_m22 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

sm11 = Slider(ax_m11, 'M11', -2.0, 2.0, valinit=initial_values['m11'])
sm12 = Slider(ax_m12, 'M12', -2.0, 2.0, valinit=initial_values['m12'])
sm21 = Slider(ax_m21, 'M21', -2.0, 2.0, valinit=initial_values['m21'])
sm22 = Slider(ax_m22, 'M22', -2.0, 2.0, valinit=initial_values['m22'])

sm11.on_changed(update)
sm12.on_changed(update)
sm21.on_changed(update)
sm22.on_changed(update)

# Add a reset button
ax_reset = plt.axes([0.05, 0.025, 0.1, 0.04], facecolor=axcolor)
button = Button(ax_reset, 'Reset', color=axcolor, hovercolor='0.975')
button.on_clicked(reset)

ax_check = plt.axes([0.05, 0.5, 0.21, 0.1], facecolor=axcolor)
show_eigen = CheckButtons(ax_check, ['Eigenvectors'], [False])
show_eigen.on_clicked(update)



plt.show()
