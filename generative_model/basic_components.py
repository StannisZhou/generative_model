from generative_model.utils import ParamsProc, Component
import numpy as np
from tqdm import tqdm
import autograd.numpy as gnp
from autograd import grad


class LineletsModel(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.add(
            'self_rooting_prob', np.ndarray,
            'The self-rooting probabilities for the two latent layers and the image layer'
        )
        proc.add(
            'image_dimension', int,
            'The dimension of the image. We are assuming the image square'
        )
        proc.add(
            'linelet_templates', list,
            'A list of the linelets templates'
        )
        proc.add(
            'cycles_perturbed', np.ndarray,
            'The perturbed distribution on the number of cycles'
        )
        proc.add(
            'n_samples_for_empirical', int,
            'The number of samples we are going to use to estimate the empirical distribution on cycles',
            int(1e5)
        )
        return proc

    @staticmethod
    def params_proc(params):
        max_cycles_list = np.array([np.sum(template) for template in params['linelet_templates']])
        params['max_cycles'] = int(np.max(max_cycles_list))

    @staticmethod
    def params_test(params):
        assert params['self_rooting_prob'].shape == (3,)
        for p in params['self_rooting_prob']:
            assert p >= 0 and p <= 1

        for template in params['linelet_templates']:
            assert type(template) ==  np.ndarray
            assert template.shape == (params['image_dimension'], params['image_dimension'])
            assert np.sum(template == 0) + np.sum(template == 1) == params['image_dimension']**2

        assert params['cycles_perturbed'].shape == (params['max_cycles'] + 1,)
        assert np.isclose(np.sum(params['cycles_perturbed']), 1)

    def __init__(self, params):
        super().__init__(params)
        self.set_cycles_empirical_distribution()
        self.log_prob_gradients = grad(linelet_model_log_prob)

    def set_cycles_empirical_distribution(self):
        self.logger.info('Getting empirical distribution for the number of cycles')
        n_samples = self.params['n_samples_for_empirical']
        n_cycles_samples = np.zeros(n_samples, dtype=int)
        templates = self.params['linelet_templates']
        n_templates = len(self.params['linelet_templates'])
        indices = np.random.randint(0, n_templates, (n_samples, 2))
        for ii in tqdm(range(n_samples)):
            n_cycles_samples[ii] = np.sum(templates[indices[ii, 0]] * templates[indices[ii, 1]])

        max_cycles = self.params['max_cycles']
        assert np.max(n_cycles_samples) <= max_cycles
        cycles_empirical = np.zeros(max_cycles + 1)
        for ii in tqdm(range(max_cycles + 1)):
            cycles_empirical[ii] = np.sum(n_cycles_samples == ii)

        cycles_empirical /= np.sum(cycles_empirical)
        cycles_empirical *= self.params['self_rooting_prob'][0]
        cycles_empirical[0] += (1 - self.params['self_rooting_prob'][0])
        self.cycles_empirical = cycles_empirical

    def get_log_prob(self, state, layers_to_update):
        params = {
            'self_rooting_prob': self.params['self_rooting_prob'],
            'linelet_templates': self.params['linelet_templates'],
            'cycles_empirical': self.cycles_empirical,
            'cycles_perturbed': self.params['cycles_perturbed']
        }
        return linelet_model_log_prob(state, params, layers_to_update)

    def get_log_prob_gradients(self, state, layers_to_update):
        params = {
            'self_rooting_prob': self.params['self_rooting_prob'],
            'linelet_templates': self.params['linelet_templates'],
            'cycles_empirical': self.cycles_empirical,
            'cycles_perturbed': self.params['cycles_perturbed']
        }
        return self.log_prob_gradients(state, params, layers_to_update)


def linelet_model_log_prob(unormalized_state, params, layers_to_update):
    """linelet_model_log_prob
    Function for calculating the log probability of the linelet model for a given state

    Parameters
    ----------

    unormalized_state : dict
        A dictionary containing the state of the model. Allowed keys include 'top', 'mid' and 'img'.
        For layers in layers_to_update, we are going to apply a sigmoid renormalization. For all other layers,
        we have a pmf as states.
        unormalized_state['top']: np array of shape (2,), which contains the state for the top layer.
        unormalized_state['mid']: list of two np arrays, each one of shape (n_templates + 1,).
        unormalized_state['img']: np array of shape (img_dim**2, 2), which represents the image.
    params : dict
        params['self_rooting_prob']: self-rooting probabilities
        params['linelet_templates']: templates for the linelets
        params['cycles_empirical']: empirical distribution for cycles from the Markov backbone
        params['cycles_perturbed']: the perturbed distribution for the number of cycles

    Returns
    -------

    log_prob: The log probability for the state

    """
    state = {}
    if 'top' in layers_to_update:
        state['top'] = gnp.exp(unormalized_state['top']) / gnp.sum(gnp.exp(unormalized_state['top']))
    else:
        state['top'] = unormalized_state['top']

    if 'mid' in layers_to_update:
        state['mid'] = []
        for ii in range(2):
            state['mid'].append(
                gnp.exp(unormalized_state['mid'][ii]) / gnp.sum(gnp.exp(unormalized_state['mid'][ii]))
            )
    else:
        state['mid'] = unormalized_state['mid']

    if 'img' in layers_to_update:
        state['img'] = gnp.exp(unormalized_state['img']) / gnp.sum(
            gnp.exp(unormalized_state['img']), axis=1, keepdims=True
        )
    else:
        state['img'] = unormalized_state['img']

    n_templates = len(params['linelet_templates'])
    img_dim = int(gnp.sqrt(state['img'].shape[0]))
    top_prob = gnp.array([1 - params['self_rooting_prob'][0], params['self_rooting_prob'][0]])
    top_log_prob = calc_log_prob(top_prob, state['top'])
    mid_self_rooting_prob = gnp.concatenate((
        gnp.array([1 - params['self_rooting_prob'][1]]),
        params['self_rooting_prob'][1] * gnp.ones(n_templates) / n_templates
    ))
    mid_parent_prob = gnp.concatenate((
        gnp.zeros(1), gnp.ones(n_templates) / n_templates
    ))
    mid_log_prob = 0
    for mid_state in state['mid']:
        mid_log_prob += state['top'][0] * calc_log_prob(mid_self_rooting_prob, mid_state)
        mid_log_prob += state['top'][1] * calc_log_prob(mid_parent_prob, mid_state)

    img_self_rooting_prob = gnp.array([1 - params['self_rooting_prob'][2], params['self_rooting_prob'][2]])
    img_parent_prob = gnp.array([0, 1])
    img_without_parent_list = [gnp.zeros((img_dim, img_dim)) for _ in range(2)]
    for mm, mid_state in enumerate(state['mid']):
        for ii in range(n_templates):
            img_without_parent_list[mm] += mid_state[ii + 1] * params['linelet_templates'][ii]

        img_without_parent_list[mm] = 1 - img_without_parent_list[mm]

    img_without_parent = (img_without_parent_list[0] * img_without_parent_list[1]).flatten()
    img_with_parent = 1 - img_without_parent
    img_log_prob = 0
    img_log_prob += calc_log_prob(
        gnp.tile(img_self_rooting_prob.reshape((1, -1)), (state['img'].shape[0], 1)),
        img_without_parent.reshape((-1, 1)) * state['img']
    )
    img_log_prob += calc_log_prob(
        gnp.tile(img_parent_prob.reshape((1, -1)), (state['img'].shape[0], 1)),
        img_with_parent.reshape((-1, 1)) * state['img']
    )
    template_matrix = gnp.stack([template.flatten() for template in params['linelet_templates']])
    n_cycles = gnp.dot(template_matrix, template_matrix.T)
    mixture_prob_matrix = state['top'][1] * gnp.dot(
        state['mid'][0][1:].reshape((-1, 1)), state['mid'][1][1:].reshape((1, -1))
    )
    n_cycles_mixture_prob = []
    for ii in range(len(params['cycles_empirical'])):
        n_cycles_mixture_prob.append(gnp.sum(mixture_prob_matrix[n_cycles == ii]))

    n_cycles_mixture_prob[0] += (1 - gnp.sum(mixture_prob_matrix))
    n_cycles_mixture_prob = gnp.array(n_cycles_mixture_prob)
    assert gnp.isclose(gnp.sum(n_cycles_mixture_prob), 1)
    cycles_log_prob = calc_log_prob(
        params['cycles_perturbed'], n_cycles_mixture_prob
    ) - calc_log_prob(params['cycles_empirical'], n_cycles_mixture_prob)
    log_prob = top_log_prob + mid_log_prob + img_log_prob + cycles_log_prob
    return log_prob


def calc_log_prob(prob, state):
    state = state[prob > 0]
    prob = prob[prob > 0]
    return gnp.sum(state * gnp.log(prob))
