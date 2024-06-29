import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# Define the initial transformation matrix and its initial slider values in order
initial_values = [
    ('m11', 1), ('m12', 0), ('m13', 0),
    ('m21', 0), ('m22', 1), ('m23', 0),
    ('m31', 0), ('m32', 0), ('m33', 1)
]

initial_values_reversed = initial_values[::-1] # for showing sliders in order m11, m12, ... , m32, m33
previous_eigen = []
colors = ['yellow', 'green', 'purple']

# Generate a grid in the 3D space
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
z = np.linspace(-1, 1, 5)
X, Y, Z = np.meshgrid(x, y, z)
points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

# Setup the figure and axes for the 3D plot
fig = plt.figure(figsize=(10, 11))
ax = fig.add_subplot(111, projection='3d')
ax.set_position([0.2, 0.3, 0.65, 0.75])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_box_aspect([1, 1, 1])
ax.set_title('3D Linear Transformation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
det_text = ax.text(-1, 0, 4, 'Determinant: 1')
legend = ax.legend()

# Draw the initial grid lines
lines = []
for i in range(len(x)):
    for j in range(len(y)):
        line, = ax.plot(X[:, i, j], Y[:, i, j], Z[:, i, j], 'gray', alpha=0.7)
        lines.append(line)
        line, = ax.plot(X[i, :, j], Y[i, :, j], Z[i, :, j], 'gray', alpha=0.7)
        lines.append(line)
        line, = ax.plot(X[i, j, :], Y[i, j, :], Z[i, j, :], 'gray', alpha=0.7)
        lines.append(line)

transformed_lines = []
for _ in range(len(lines)):
    line, = ax.plot([], [], [], 'red', alpha=0.5)
    transformed_lines.append(line)

# Function to apply transformation matrix to points
def apply_transformation(matrix, points):
    return np.dot(matrix, points.reshape(3, -1))


def plot_eigen(matrix, show):
    global previous_eigen
    global legend
    # Clear previous arrows
    for eigen in previous_eigen:
        eigen.remove()
    previous_eigen = []

    if show:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        # Plot the eigenvectors
        for i in range(len(eigenvalues)):
            start_point = [0, 0, 0]
            end_point = eigenvectors[:, i] * eigenvalues[i]
            eigen = ax.quiver(start_point[0], start_point[1], start_point[2],  end_point[0], end_point[1], end_point[2],
                              color=colors[i],
                              label=f'Eigenvector {i + 1} with $\lambda={eigenvalues[i]:.3f}$')
            previous_eigen.append(eigen)
        ax.legend(loc='upper left')
    else:
        legend.remove()


# Update the figure based on transformation matrix
def update(val):
    show = show_eigen.get_status()[0]
    m = np.array([slider.val for slider in sliders[::-1]])
    transform_matrix = m.reshape(3, 3)
    transformed_points = apply_transformation(transform_matrix, points).reshape(3, len(x), len(y), len(z))
    plot_eigen(transform_matrix, show)
    det = np.linalg.det(transform_matrix)
    det_text.set_text(f'Determinant: {det:.3f}')

    k = 0
    for i in range(len(x)):
        for j in range(len(y)):
            transformed_lines[k].set_data(transformed_points[0, :, i, j], transformed_points[1, :, i, j])
            transformed_lines[k].set_3d_properties(transformed_points[2, :, i, j])
            k += 1
            transformed_lines[k].set_data(transformed_points[0, i, :, j], transformed_points[1, i, :, j])
            transformed_lines[k].set_3d_properties(transformed_points[2, i, :, j])
            k += 1
            transformed_lines[k].set_data(transformed_points[0, i, j, :], transformed_points[1, i, j, :])
            transformed_lines[k].set_3d_properties(transformed_points[2, i, j, :])
            k += 1
    fig.canvas.draw_idle()

# Reset function
def reset(event):
    for slider in sliders:
        slider.reset()

# Create sliders for each matrix element
axcolor = 'lightgoldenrodyellow'
slider_height = 0.02
slider_width = 0.65
slider_start = 0.05  # Starting y position for sliders
slider_interval = 0.03
sliders = []
for idx, (key, initial) in enumerate(initial_values_reversed):
    ax_slider = plt.axes([0.25, 0.1 + idx * slider_interval, slider_width, slider_height])
    slider = Slider(ax_slider, key, -2.0, 2.0, valinit=initial)
    slider.on_changed(update)
    sliders.append(slider)

# Add a reset button
ax_reset = plt.axes([0.05, 0.1, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
button_reset.on_clicked(reset)

ax_check = plt.axes([0.05, 0.5, 0.18, 0.04], facecolor=axcolor)
show_eigen = CheckButtons(ax_check, ['Eigenvectors'], [False])
show_eigen.on_clicked(update)


plt.show()
