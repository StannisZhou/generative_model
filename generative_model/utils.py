import pprint
import logging
import pytz
from datetime import datetime
import pickle

class ParamsProc(object):
    def __init__(self):
        self.keys = set()
        self.required_keys = set()
        self.optional_params = {}
        self.params_types = {}
        self.params_help = {}
        self.add(
            'root_folder', str,
            'The path to the root folder in which we are running the experiments',
            None
        )
        self.add(
            'output_identifier', str,
            'The output identifier, which we are going to use as the folder name for storing all the outputs',
            None
        )

    def add(self, key, param_type, help_info, default=None):
        if key in set(['root_folder', 'output_identifier']):
            self.optional_params[key] = default
        else:
            if default is None:
                self.required_keys.add(key)
            else:
                assert type(default) is param_type
                self.optional_params[key] = default

        self.keys.add(key)
        self.params_types[key] = param_type
        self.params_help[key] = help_info

    def update(self, proc, excluded_params=[]):
        if not hasattr(self, 'keys'):
            self.keys = {}

        keys = proc.keys.difference(set(excluded_params))
        required_keys = proc.keys.difference(set(excluded_params))
        optional_params = {
            key: proc.optional_params[key] for key in proc.optional_params if key not in excluded_params
        }
        params_types = {
            key: proc.params_types[key] for key in proc.params_types if key not in excluded_params
        }
        params_help = {
            key: proc.params_help[key] for key in proc.params_help if key not in excluded_params
        }
        assert len(self.keys.intersection(keys).difference(set(['root_folder', 'output_identifier']))) == 0
        self.keys.update(keys)
        self.required_keys.update(required_keys)
        self.optional_params.update(optional_params)
        self.params_types.update(params_types)
        self.params_help.update(params_help)

    def update_multiple(self, proc_list, excluded_params=[]):
        if not hasattr(self, 'keys'):
            self.keys = {}

        keys = set()
        required_keys = set()
        optional_params = {}
        params_types = {}
        params_help = {}
        for proc in proc_list:
            keys.update(proc.keys.difference(set(excluded_params)))
            required_keys.update(proc.keys.difference(set(excluded_params)))
            optional_params.update({
                key: proc.optional_params[key] for key in proc.optional_params if key not in excluded_params
            })
            params_types.update({
                key: proc.params_types[key] for key in proc.params_types if key not in excluded_params
            })
            params_help.update({
                key: proc.params_help[key] for key in proc.params_help if key not in excluded_params
            })

        assert len(self.keys.intersection(keys).difference(set(['root_folder', 'output_identifier']))) == 0
        self.keys.update(keys)
        self.required_keys.update(required_keys)
        self.optional_params.update(optional_params)
        self.params_types.update(params_types)
        self.params_help.update(params_help)

    def get_empty_params(self):
        params = {}
        for key in self.required_keys:
            params[key] = None

        params.update(self.optional_params)
        params['keys_to_proc'] = []
        return params

    def process_params(self, params, params_proc=None, params_test=None):
        assert self.required_keys.issubset(set(params.keys())), 'Required: {}, given: {}, missing: {}'.format(
            self.required_keys, params.keys(), self.required_keys.difference(set(params.keys()))
        )
        for key in self.optional_params:
            if key not in params:
                params[key] = self.optional_params[key]

        assert set(params.keys()).issubset(self.keys), 'Given: {}, allowed: {}, missing: {}'.format(
            params.keys(), self.keys, set(params.keys()).difference(self.keys)
        )
        for key in params:
            assert (
                type(params[key]) is self.params_types[key] or params[key] is None
            ), 'Key: {}, given type: {}, required type: {}'.format(
                key, type(params[key]), self.params_types[key]
            )

        if params_proc is not None:
            params_proc(params)

        if params_test is not None:
            params_test(params)

        return params


class Component(object):
    @staticmethod
    def get_proc():
        raise NotImplementedError

    @staticmethod
    def params_proc():
        raise NotImplementedError

    @staticmethod
    def params_test():
        raise NotImplementedError

    def __init__(self, params):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger = logging.getLogger(type(self).__name__)
        logger.addHandler(console)
        self.logger = logger
        proc = self.get_proc()
        self.logger.info('\n\nInitializing class {}, realized parameters:\n'.format(type(self).__name__))
        self.params = proc.process_params(params, self.params_proc, self.params_test)
        self.logger.info(pprint.pformat(self.params))
        if self.params['root_folder'] is not None and self.params['output_identifier'] is not None:
            timezone = pytz.timezone('US/Eastern')
            fmt = '%Y-%m-%d_%H-%M-%S_%Z'
            output_identifier = datetime.now(timezone).strftime(fmt)
            realized_params_fname = '{}/output/{}/{}_realized_params_{}.pkl'.format(
                self.params['root_folder'], self.params['output_identifier'],
                type(self).__name__, output_identifier
            )
            with open(realized_params_fname, 'wb') as f:
                pickle.dump(self.params, f)

    def construct_target_class(self, target_class, optional_params={}):
        target_proc = target_class.get_proc()
        params = {
            key: self.params[key] for key in set(self.params.keys()).intersection(
                target_class.get_proc().keys
            )
        }
        params.update(optional_params)
        obj = target_class(params)
        return obj
