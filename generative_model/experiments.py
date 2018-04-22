from generative_model.utils import ParamsProc, Component
from generative_model.basic_components import LineletsModel
import pickle
import copy
import numpy as np


class LineletsModelGradientAscent(Component):
    @staticmethod
    def get_proc():
        proc = ParamsProc()
        proc.update(LineletsModel.get_proc())
        proc.add(
            'learning_rate', float,
            'The learning rate we are going to use'
        )
        proc.add(
            'initial_state', dict,
            'The initial state we are going to start the gradient ascent'
        )
        proc.add(
            'layers_to_update', list,
            'A list of the layers we are going to update',
            ['mid', 'img']
        )
        proc.add(
            'n_iters', int,
            'The number of iterations we are going to use to estimate the states'
        )
        return proc

    @staticmethod
    def params_proc(params):
        params['results_fname'] = '{}/output/{}/results.pkl'.format(params['root_folder'], params['output_identifier'])

    @staticmethod
    def params_test(params):
        assert set(params['layers_to_update']).issubset(set(['top', 'mid', 'img']))
        if 'top' not in params['layers_to_update']:
            assert np.sum(params['initial_state']['top']) == 1

        if 'mid' not in params['layers_to_update']:
            for ii in range(2):
                assert np.sum(params['initial_state']['mid'][ii]) == 1

        if 'img' not in params['layers_to_update']:
            assert np.all(np.sum(params['initial_state']['img'], axis=1) == np.ones(params['initial_state']['img'].shape[0]))

    def __init__(self, params):
        super().__init__(params)
        self.linelets_model = self.construct_target_class(LineletsModel)

    def run_experiments(self):
        training_evolution = []
        log_prob_evolution = np.zeros(self.params['n_iters'] + 1)
        state = self.params['initial_state']
        log_prob = self.linelets_model.get_log_prob(state, self.params['layers_to_update'])
        self.logger.info('Initial log prob: {}'.format(log_prob))
        log_prob_evolution[0] = log_prob
        training_evolution.append(copy.deepcopy(state))
        for ii in range(self.params['n_iters']):
            state_grad = self.linelets_model.get_log_prob_gradients(state, self.params['layers_to_update'])
            self.update_state(state, state_grad)
            log_prob = self.linelets_model.get_log_prob(state, self.params['layers_to_update'])
            self.logger.info('Iteration #{} log prob: {}'.format(ii + 1, log_prob))
            log_prob_evolution[ii + 1] = log_prob
            training_evolution.append(copy.deepcopy(state))

        results = {
            'log_prob_evolution': log_prob_evolution,
            'training_evolution': training_evolution
        }
        with open(self.params['results_fname'], 'wb') as f:
            pickle.dump(results, f)

    def update_state(self, state, state_grad):
        for layer in self.params['layers_to_update']:
            if layer == 'mid':
                for ii in range(2):
                    state['mid'][ii] += self.params['learning_rate'] * state_grad['mid'][ii]
            else:
                state[layer] += self.params['learning_rate'] * state_grad[layer]
