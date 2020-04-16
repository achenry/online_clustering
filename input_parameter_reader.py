import configparser

class InputParameterReader:

    def __init__(self):
        self.data_type = None
        self.batch_size = None
        self.init_batch_size = None
        self.init_num_clusters = None
        # select tolerance
        self.tol = None
        # select max iterations
        self.max_iter = None
        self.gravitational_const = None  # * 10**(-11)
        self.k_change_threshold = None
        self.algorithm = None
        self.quick_test = None
        self.test_name = None
        self.test_params = None

    def read_input_parameters(self):
        config = configparser.ConfigParser()
        config.read('input_params.ini')

        try:
            self.data_type = int(config['PARAMETERS']['data_type'])
            if not any([self.data_type == dt for dt in [0, 1]]):
                raise ValueError('data_type input parameter must be 0 or 1')
        except Exception as e:
            print(e)

        try:
            self.batch_size = int(config['PARAMETERS']['batch_size'])
            if self.batch_size < 1:
                raise ValueError('batch_size input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.init_batch_size = int(config['PARAMETERS']['init_batch_size'])
            if self.init_batch_size < 1:
                raise ValueError('init_batch_size input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.init_num_clusters = int(config['PARAMETERS']['init_num_clusters'])
            if self.init_batch_size < 2:
                raise ValueError('init_batch_size input parameter must be greater than 1')
        except Exception as e:
            print(e)

        try:
            self.tol = float(config['PARAMETERS']['tol'])
            if self.tol <= 0:
                raise ValueError('tol input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.max_iter = int(config['PARAMETERS']['max_iter'])
            if self.max_iter < 1:
                raise ValueError('max_iter input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.gravitational_const = float(config['PARAMETERS']['gravitational_const'])  # * 10**(-11)
            if self.gravitational_const < 0:
                raise ValueError('gravitational_const input parameter must be greater or equal to 0')
        except Exception as e:
            print(e)

        try:
            self.k_change_threshold = float(config['PARAMETERS']['k_change_threshold'])
            if self.k_change_threshold <= 0:
                raise ValueError('k_change_threshold input parameter must be greater than 0')
        except Exception as e:
            print(e)

        try:
            self.algorithm = int(config['PARAMETERS']['algorithm'])
            if not any([self.algorithm == a for a in [0, 1]]):
                raise ValueError('algorithm input parameter must be 0 or 1')
        except Exception as e:
            print(e)

        try:
            self.quick_test = bool(config['PARAMETERS']['data_type'])
        except Exception as e:
            print(e)

        try:
            self.test_name = config['PARAMETERS']['data_type']
        except Exception as e:
            print(e)

        self.test_params = dict(config['PARAMETERS'])
