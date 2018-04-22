import numpy as np
from skimage.transform import rotate
from scipy.stats import norm
import matplotlib.pyplot as plt
from generative_model.experiments import LineletsModelGradientAscent
import os
import shutil
import logging
import pickle


ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
output_folder = '{}/output'.format(ROOT_FOLDER)
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.mkdir(output_folder)


# Set up linelet templates
image_dimension = 4
linelet_shape = (1, 3)
linelet_prototype = np.zeros((linelet_shape[1], linelet_shape[1]))
linelet_prototype[1, :] = 1
n_rotations = 4
rotation_angles = np.linspace(0, 180, n_rotations, endpoint=False)
linelet_rotations = np.zeros((n_rotations, linelet_shape[1], linelet_shape[1]))
for ii in range(n_rotations):
    linelet_rotations[ii] = rotate(linelet_prototype, angle=rotation_angles[ii], order=0, preserve_range=True)

n_loc_per_side = image_dimension - linelet_shape[1] + 1
linelet_templates = np.zeros((n_loc_per_side, n_loc_per_side, n_rotations, image_dimension, image_dimension))
for ii in range(n_loc_per_side):
    for jj in range(n_loc_per_side):
        for kk in range(n_rotations):
            linelet_templates[ii, jj, kk, ii:(ii + linelet_shape[1]), jj:(jj + linelet_shape[1])] = linelet_rotations[kk]

linelet_templates = list(linelet_templates.reshape((-1, image_dimension, image_dimension)))

# Set cycles perturbation term
max_cycles_list = np.array([np.sum(template) for template in linelet_templates])
max_cycles = int(np.max(max_cycles_list))
cycles_perturbed = np.array([0.01, 0.01, 0.97, 0.01])

# Set up the common parameters
base_params = {
    'n_iters': int(1e3),
    'learning_rate': 5.,
    'image_dimension': image_dimension,
    'cycles_perturbed': cycles_perturbed,
    'self_rooting_prob': np.array([0.5, 0.1, 0.01]),
    'n_samples_for_empirical': int(5e4),
    'linelet_templates': linelet_templates,
    'root_folder': ROOT_FOLDER
}

with open('{}/base_params.pkl'.format(output_folder), 'wb') as f:
    pickle.dump(base_params, f)

# Experiments with both top layer and image given
initial_state = {
    'top': np.array([0, 1], dtype=float),
    'mid': [np.random.rand(len(linelet_templates) + 1), np.random.rand(len(linelet_templates) + 1)],
    'img': np.stack((np.ones(image_dimension**2), np.zeros(image_dimension**2)), axis=1)
}
initial_state['img'][4:8, :] = np.stack((np.zeros(4), np.ones(4)), axis=1)
layers_to_update = ['mid']
output_identifier = 'mid_only'

params = base_params.copy()
params.update({
    'initial_state': initial_state,
    'layers_to_update': layers_to_update,
    'output_identifier': output_identifier
})

## Set up logging and run experiments
logs_fname = '{}/{}/logs.txt'.format(output_folder, output_identifier)
logs_level = logging.INFO
if not os.path.exists(os.path.dirname(logs_fname)):
    os.makedirs(os.path.dirname(logs_fname))

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logs_level,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%d%m%y-%H%M%S',
    filename=logs_fname,
    filemode='w'
)
obj = LineletsModelGradientAscent(params)
obj.run_experiments()

# Experiment with only top layer given
initial_state = {
    'top': np.array([0, 1], dtype=float),
    'mid': [np.random.rand(len(linelet_templates) + 1), np.random.rand(len(linelet_templates) + 1)],
    'img': np.random.rand(image_dimension**2, 2)
}
layers_to_update = ['mid', 'img']
output_identifier = 'mid_and_img'

params = base_params.copy()
params.update({
    'initial_state': initial_state,
    'layers_to_update': layers_to_update,
    'output_identifier': output_identifier
})

## Set up logging and run experiments
logs_fname = '{}/{}/logs.txt'.format(output_folder, output_identifier)
logs_level = logging.INFO
if not os.path.exists(os.path.dirname(logs_fname)):
    os.makedirs(os.path.dirname(logs_fname))

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logs_level,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%d%m%y-%H%M%S',
    filename=logs_fname,
    filemode='w'
)
obj = LineletsModelGradientAscent(params)
obj.run_experiments()
