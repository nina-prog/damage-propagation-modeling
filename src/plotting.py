import matplotlib.pyplot as plt
import time


def plot_sensor_measures_from_motor(config, motor_id, data, name_of_dataset, show=True, save=False):
    """ This method plots the sensor measures from a motor with the given motor_id.

    :param config: A configuration file with the paths for saving the plot.
    :param motor_id: The motor id.
    :param data: The dataset with the sensor measures of the motor.
    :param name_of_dataset: The name of the dataset. Only relevant for saving the plot.
    :param show: Whether to show the plot.
    :param save: Whether to save the plot.
    """
    sensor_measure_columns_names = list(filter(lambda x: x.startswith('Sensor'), train_data.keys()))
    number_of_columns = len(sensor_measure_columns_names)
    fig = plt.figure(figsize=(15,2.5*len(sensor_measure_columns_names)))
    Dataframe_id = data[data["UnitNumber"] == motor_id]
    for i,v in enumerate(range(number_of_columns)):
        a = plt.subplot(number_of_columns,1,v+1)
        a.plot(Dataframe_id.index.values,Dataframe_id.iloc[:,v+5].values)
        a.title.set_text(sensor_measure_columns_names[v])
        plt.tight_layout()
    if show:
        plt.show()
    if save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fig.savefig(f"{config['paths']['plot_dir']}ex2_topic_dataset_{name_of_dataset}_motor_id_"
                    f"{motor_id}_{timestamp}.png")
    plt.close()


def plot_sensor_measures_from_dataset(config, data, name_of_dataset, show=True, save=False):
    """ This method plots the sensor measures from all motors from the given dataset. Furthermore, it plots the mean
    signal of the sensor measures in red.

    :param config: A configuration file with the paths for saving the plot.
    :param data: The dataset with the sensor measure
    :param name_of_dataset: The name of the dataset. Only relevant for saving the plot.
    :param show: Whether to show the plot.
    :param save: Whether to save the plot.
    """
    sensor_measure_columns_names = list(filter(lambda x: x.startswith('Sensor'), data.keys()))
    number_of_columns = len(sensor_measure_columns_names)
    fig = plt.figure(figsize=(15,2.5*len(sensor_measure_columns_names)))
    for i,v in enumerate(range(number_of_columns)):
        a = plt.subplot(number_of_columns,1,v+1)
        min_rul = data.groupby("UnitNumber")["RUL"].max().min()
        a.plot(range(min_rul), data[data["RUL"] <= min_rul].groupby("RUL")[sensor_measure_columns_names[v]].
               mean(), color='r')
        for motor_id in data['UnitNumber'].unique():
            a.plot(range(min_rul-1, -1, -1), data[data["UnitNumber"] == motor_id]
                                             [sensor_measure_columns_names[v]][-min_rul:], alpha=0.1, color='b')
        a.invert_xaxis()
        a.title.set_text(sensor_measure_columns_names[v])
        plt.tight_layout()
    if show:
        plt.show()
    if save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fig.savefig(f"{config['paths']['plot_dir']}ex2_topic_dataset_{name_of_dataset}_{timestamp}.png")
    plt.close()

