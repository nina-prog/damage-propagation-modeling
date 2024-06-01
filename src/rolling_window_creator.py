"""This file contains the RollingWindowDatasetCreator class, which is responsible for creating rolling windows from
the dataset. The class also extracts features from the rolling windows. The class is used to create the training and
testing datasets for the RUL prediction task."""
import pandas as pd

from tsfresh.feature_extraction import extract_features, feature_calculators, MinimalFCParameters, ComprehensiveFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.dataframe_functions import roll_time_series

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def calculate_RUL(data: pd.DataFrame, time_column: str, group_column: str, clipping_value_of_RUL: int = None) \
        -> pd.DataFrame:
    """Generate the remaining useful life (RUL) for the dataset. The RUL is the number of cycles before the machine
    fails. RUL at failure is 1.

    :param group_column: The name of the column that identifies the units.
    :type group_column: str
    :param time_column: The name of the time column based on which the RUL will be calculated.
    :type time_column: str
    :param data: The dataset.
    :type data: pd.DataFrame
    :param clipping_value_of_RUL: Every higher rul will be clipped to this value.
    :type clipping_value_of_RUL: int

    :return: The dataset with the RUL column.
    :rtype: pd.DataFrame
    """

    data = data.copy()
    data['RUL'] = data.groupby(group_column)[time_column].transform("max") - data[time_column]
    # adding one because the current cycle also counts (RUL at failure is 1)
    data['RUL'] = data['RUL'] + 1

    if clipping_value_of_RUL is not None:
        data['RUL'] = data['RUL'].apply(lambda x: clipping_value_of_RUL if x > clipping_value_of_RUL else x)
    logger.debug("RUL generated successfully.")

    return data


class RollingWindowDatasetCreator:
    """The RollingWindowDatasetCreator class is responsible for creating rolling windows from the dataset. The class
    also extracts features from the rolling windows."""
    def __init__(self, column_id: str = "UnitNumber", column_sort: str = "Cycle", max_timeshift: int = 20,
                 min_timeshift: int = 0, feature_extraction_mode: str = 'minimal', feature_list: list = ['sum_values', 'abs_energy', 'fft_coefficient', 'benford_correlation', 'permutation_entropy', 'median', 'first_location_of_maximum', 'percentage_of_reoccurring_values_to_all_values', 'sum_of_reoccurring_values', 'sum_of_reoccurring_data_points', 'ratio_value_number_to_time_series_length', 'change_quantiles', 'maximum', 'absolute_maximum', 'quantile', 'binned_entropy', 'agg_linear_trend', 'cwt_coefficients',  '']) -> None:
        """Initialize the RollingWindowDatasetCreator class.

        :param column_id: The name of the column that identifies the units.
        :type column_id: str
        :param column_sort: The name of the column based on which the data will be sorted.
        :type column_sort: str
        :param max_timeshift: The maximum window size.
        :type max_timeshift: int
        :param min_timeshift: The minimum window size.
        :type min_timeshift: int
        :param feature_extraction_mode: The mode of feature extraction. It can be 'minimal', 'all', 'efficient' or 'custom'.
        :type feature_extraction_mode: str
        :param feature_list: For defining a custom feature list in the feature extraction mode "custom"
        :type  feature_list: list[str]
        """
        self.column_id = column_id
        self.column_sort = column_sort
        self.max_timeshift = max_timeshift
        self.min_timeshift = min_timeshift
        self.feature_extraction_mode = feature_extraction_mode
        self.feature_list = feature_list
        self.default_fc_parameters = self._get_default_fc_parameters()


    def _get_default_fc_parameters(self):
        """Get the default feature extraction parameters based on the feature_extraction_mode.

        :return: The default feature extraction parameters.
        :rtype: dict
        """
        custom_fc_parameters = EfficientFCParameters()
        if self.feature_extraction_mode == 'custom':
            for fname in feature_calculators.__dict__.keys():
                if fname in custom_fc_parameters and not fname in self.feature_list:
                    del custom_fc_parameters[fname]

        mode_mapping = {
            'minimal': MinimalFCParameters(),
            'all': ComprehensiveFCParameters(),
            'efficient': EfficientFCParameters(),
            'custom': custom_fc_parameters,
        }

        if isinstance(self.feature_extraction_mode, dict):
            return self.feature_extraction_mode
        else:
            default_fc_parameters = mode_mapping.get(self.feature_extraction_mode)
            if default_fc_parameters is None:
                raise ValueError("feature_extraction_mode must be 'minimal', 'all', 'efficient' or 'custom'.")
            return default_fc_parameters

    def validate_window_sizes(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Validate the window sizes. The minimum window size must be greater than 0, the maximum window size must be
        greater than the minimum window size, and the maximum window size must be less than or equal to the minimum
        number of cycles in the dataset.

        :param train_data: The training dataset.
        :type train_data: pd.DataFrame
        :param test_data: The testing dataset.
        :type test_data: pd.DataFrame

        :raises ValueError: If the window sizes are invalid.
        """
        min_cycles_test = test_data.groupby(self.column_id)[self.column_sort].count().min()
        min_cycles_train = train_data.groupby(self.column_id)[self.column_sort].count().min()
        min_cycles_total = min(min_cycles_test, min_cycles_train)

        if not (0 < self.min_timeshift < self.max_timeshift <= min_cycles_total):
            raise ValueError(f"Invalid window sizes: min_window_size={self.min_timeshift}, max_window_size={self.max_timeshift}. "
                             f"Conditions: 0 < min_window_size < max_window_size <= {min_cycles_total}.")

    def _process_data(self, data: pd.DataFrame, data_type: str, test_RUL_data: pd.DataFrame = None) -> tuple:
        """Create rolling windows and extract features from the data.

        :param data: The dataset.
        :type data: pd.DataFrame
        :param data_type: The type of data. It can be 'train' or 'test'.
        :type data_type: str
        :param test_RUL_data: The RUL data for the test data.
        :type test_RUL_data: pd.DataFrame

        :return: The features and the target.
        :rtype: tuple
        """
        logger.info(f"Creating rolling windows for {data_type} data...")
        rolled_data = roll_time_series(data, column_id=self.column_id, column_sort=self.column_sort,
                                       max_timeshift=self.max_timeshift, min_timeshift=self.min_timeshift)
        if data_type == 'test':
            rolled_data = rolled_data.groupby(self.column_id).tail(self.max_timeshift)

        logger.info(f"Extracting features for {data_type} data...")
        X = extract_features(rolled_data.drop([self.column_id], axis=1),
                             column_id="id", column_sort=self.column_sort,
                             default_fc_parameters=self.default_fc_parameters,
                             impute_function=impute, show_warnings=False)
        X.index = X.index.rename([self.column_id, self.column_sort])

        if data_type == 'train':
            logger.info("Calculating target for train data...")
            data_rul = calculate_RUL(data=data, time_column=self.column_sort, group_column=self.column_id)
            y = data_rul.set_index([self.column_id, self.column_sort]).sort_index().RUL.to_frame()
            y = y[y.index.isin(X.index)]
            X = X[X.index.isin(y.index)]
        else:
            y = test_RUL_data
            y.index = X.index

        return X, y

    def create_rolling_windows_datasets(self, train_data: pd.DataFrame, test_data: pd.DataFrame, test_RUL_data: pd.DataFrame) -> tuple:
        """Create the rolling windows datasets.

        :param train_data: The training dataset.
        :type train_data: pd.DataFrame
        :param test_data: The testing dataset.
        :type test_data: pd.DataFrame
        :param test_RUL_data: The RUL data for the test data.
        :type test_RUL_data: pd.DataFrame

        :return: The training and testing datasets.
        :rtype: tuple
        """
        self.validate_window_sizes(train_data, test_data)
        X_train, y_train = self._process_data(train_data, 'train')
        X_test, y_test = self._process_data(test_data, 'test', test_RUL_data)

        logger.info("Datasets created successfully.")
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of y_train: {y_train.shape}")
        logger.info(f"Shape of X_test: {X_test.shape}")
        logger.info(f"Shape of y_test: {y_test.shape}")

        return X_train, y_train, X_test, y_test
