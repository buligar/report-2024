import nengo
import numpy as np
import matplotlib.pyplot as plt
from nengo_extras.gui import image_display_function
from PIL import Image
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0  # Нормализация данных

image_train_filt = []
input_nbr = 20
pixel = 12
halfpixel = int(pixel/2)
def convert_mnist_28x28_to_12x12(image_array):
    image = Image.fromarray(image_array)
    image_resized = image.resize((pixel, pixel), Image.BICUBIC)
    image_resized_array = np.array(image_resized)
    return image_resized_array

image_train_resized = np.array([convert_mnist_28x28_to_12x12(img) for img in x_train])

for i in range(0, input_nbr):
    image_train_filt.append(image_train_resized[i])

def encode_image(image):
    return image.flatten()

def decode_image(encoded_image, dimensions):
    return encoded_image.reshape(dimensions)

image_shape = (pixel, pixel)
input_dimensions = image_shape[0] * image_shape[1]
input_display_time = 1
image_train_left_black = []

for i in range(0, input_nbr):
    left_half = np.reshape(image_train_filt[i], (pixel, pixel))[:, :halfpixel]
    right_half_black = np.zeros((pixel, halfpixel))
    image_left_black = np.hstack((left_half, right_half_black))
    image_train_left_black.append(image_left_black)

with nengo.Network() as model:
    full_input_node = nengo.Node(
        nengo.processes.PresentInput(image_train_filt, input_display_time),
        label="Input"
    )
    partial_input_node = nengo.Node(
        nengo.processes.PresentInput(image_train_left_black, input_display_time),
        label="Input",
    )
    pre_args = dict(
        encoders=nengo.dists.ScatteredHypersphere(surface=True),
        intercepts=nengo.dists.Uniform(-1, 1),
        max_rates=nengo.dists.Uniform(200, 400),
        eval_points=nengo.dists.ScatteredHypersphere()
    )
    input_ens = nengo.Ensemble(n_neurons=3000, dimensions=input_dimensions, **pre_args)
    neurons = nengo.Ensemble(n_neurons=3000, dimensions=input_dimensions)
    output = nengo.Node(size_in=input_dimensions)
    
    nengo.Connection(partial_input_node, input_ens)
    conn = nengo.Connection(input_ens, neurons, synapse=None)
    nengo.Connection(neurons, output, synapse=None)
    error = nengo.Ensemble(n_neurons=3000, dimensions=input_dimensions, **pre_args)
    nengo.Connection(neurons, error, synapse=None)
    nengo.Connection(full_input_node, error, transform=-1, synapse=None)
    error_node = nengo.Node(size_in=input_dimensions)
    nengo.Connection(error, error_node, synapse=None)
    inp_ens_node = nengo.Node(size_in=input_dimensions)
    nengo.Connection(input_ens, inp_ens_node, synapse=None)
    conn.learning_rule_type = nengo.PES(learning_rate=1e-4)
    nengo.Connection(error, conn.learning_rule, synapse=None)

    image_shape = (1, pixel, pixel)
    display_func = image_display_function(image_shape, offset=0, scale=255)
    display_node = nengo.Node(display_func, size_in=pixel*pixel, label='full')
    nengo.Connection(full_input_node, display_node, synapse=None)
    display_node = nengo.Node(display_func, size_in=pixel*pixel, label='part')
    nengo.Connection(partial_input_node, display_node, synapse=None)
    display_node = nengo.Node(display_func, size_in=pixel*pixel, label='error')
    nengo.Connection(error_node, display_node, synapse=None)
    display_node = nengo.Node(display_func, size_in=pixel*pixel, label='inp_ens')
    nengo.Connection(inp_ens_node, display_node, synapse=None)
    display_node = nengo.Node(display_func, size_in=pixel*pixel, label='output')
    nengo.Connection(output, display_node, synapse=0.01)
