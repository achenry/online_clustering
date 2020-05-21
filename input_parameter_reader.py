import configparser
import os

class InputParameterReader:
    """
    Class defining reader for .ini user inputs
    """

    def __init__(self):
        """
        initialise InputParameterReader object with all parameters as None
        """
        self.data_type = None
        self.batch_size = None
        self.init_batch_size = None
        self.init_num_clusters = None
        # select tolerance
        self.tol = None
        # select max iterations
        self.max_iter = None
        self.fuzziness = None
        self.gravitational_const = None
        self.time_decay_const = None
        self.algorithm = None
        self.num_samples_to_run = None
        self.window_size = None
        self.test_name = None
        self.test_params = None
        self.csv_path = None
        self.customer_ids = None
        self.feature_names = None
        self.input_params = {}

    def read_input_parameters(self):
        """
        read user-given parameters from input_params.ini and raise error is unallowed value is given
        """
        config = configparser.ConfigParser()
        config.read('input_params.ini')

        try:
            self.data_type = int(config['PARAMETERS']['data_type'])
            self.input_params['data_type'] = self.data_type
            if not any([self.data_type == dt for dt in [0, 1]]):
                raise ValueError('data_type input parameter must be 0 or 1')
        except Exception as e:
            print(e)

        try:
            self.batch_size = int(config['PARAMETERS']['batch_size'])
            self.input_params['batch_size'] = self.batch_size
            if self.batch_size < 1:
                raise ValueError('batch_size input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.init_batch_size = int(config['PARAMETERS']['init_batch_size'])
            self.input_params['init_batch_size'] = self.init_batch_size
            if self.init_batch_size < 1:
                raise ValueError('init_batch_size input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.init_num_clusters = int(config['PARAMETERS']['init_num_clusters'])
            self.input_params['init_num_clusters'] = self.init_num_clusters
            if self.init_num_clusters < 2:
                raise ValueError('init_batch_size input parameter must be greater than 1')
        except Exception as e:
            print(e)

        try:
            self.tol = float(config['PARAMETERS']['tol'])
            self.input_params['tol'] = self.tol
            if self.tol <= 0:
                raise ValueError('tol input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.max_iter = int(config['PARAMETERS']['max_iter'])
            self.input_params['max_iter'] = self.max_iter
            if self.max_iter < 1:
                raise ValueError('max_iter input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.gravitational_const = float(config['PARAMETERS']['gravitational_const'])  # * 10**(-11)
            self.input_params['gravitational_const'] = self.gravitational_const
            if self.gravitational_const < 0:
                raise ValueError('gravitational_const input parameter must be greater than or equal to 0.')
        except Exception as e:
            print(e)

        try:
            self.time_decay_const = int(config['PARAMETERS']['time_decay_const'])  # * 10**(-11)
            self.input_params['time_decay_const'] = self.time_decay_const
            if self.time_decay_const <= 0:
                raise ValueError('time_decay_const input parameter must be greater than 0.')
        except Exception as e:
            print(e)

        try:
            self.fuzziness = float(config['PARAMETERS']['fuzziness'])  # * 10**(-11)
            self.input_params['fuzziness'] = self.fuzziness
            if self.fuzziness < 1:
                raise ValueError('fuzziness input parameter must be greater or equal to 1.')
        except Exception as e:
            print(e)

        try:
            self.window_size = int(config['PARAMETERS']['window_size'])
            self.input_params['window_size'] = self.window_size
            if self.window_size <= 0:
                raise ValueError('window_size input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.alpha = float(config['PARAMETERS']['alpha'])
            self.input_params['alpha'] = self.alpha
            if self.alpha < 0 or self.alpha > 1:
                raise ValueError('alpha input parameter must be between 0 and 1')
        except Exception as e:
            print(e)

        try:
            self.algorithm = int(config['PARAMETERS']['algorithm'])
            self.input_params['algorithm'] = self.algorithm
            if not any([self.algorithm == a for a in [0, 1, 2]]):
                raise ValueError('algorithm input parameter must be 0, 1 or 2')
        except Exception as e:
            print(e)

        try:
            self.num_samples_to_run = config['PARAMETERS']['num_samples_to_run']

            if self.num_samples_to_run != 'all':
                self.num_samples_to_run = int(self.num_samples_to_run)

            self.input_params['num_samples_to_run'] = self.num_samples_to_run

        except Exception as e:
            print(e)

        try:
            self.test_name = config['PARAMETERS']['test_name']
            self.input_params['test_name'] = self.test_name
        except Exception as e:
            print(e)

        try:
            self.plotting_data_step = int(config['PARAMETERS']['plotting_data_step'])
            self.input_params['plotting_data_step'] = self.plotting_data_step
        except Exception as e:
            print(e)

        try:
            self.csv_path = config['PARAMETERS']['csv_path']
            self.input_params['csv_path'] = self.csv_path
            if not os.path.exists(self.csv_path):
                raise ValueError('csv path does not exist.')
        except Exception as e:
            print(e)

        try:
            self.customer_ids = [int(cid) for cid in config['PARAMETERS']['customer_ids'].split(',')]
            self.input_params['customer_ids'] = self.customer_ids
        except Exception as e:
            print(e)

        try:
            self.feature_names = [feat.strip() for feat in config['PARAMETERS']['feature_names'].split(',')]
            self.input_params['feature_names'] = self.feature_names
        except Exception as e:
            print(e)
