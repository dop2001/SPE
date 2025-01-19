import csv
import os.path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


def read_data(file_path, column_name='dice'):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        column_data = [float(row[column_name].split('[')[0]) for row in reader]
        column_data = column_data
    return column_data


def axis_inter(data, ext_distance=0.1):
    min_data, max_data = min(data), max(data)
    min_data, max_data = max(min_data-ext_distance, 0), min(max_data+ext_distance, 1)
    ticks = np.arange(np.floor(min_data * 10) / 10, np.ceil(max_data * 10) / 10 + 0.1, 0.1)
    return ticks


def draw(x_path, y_path, x_test_path, y_test_path, curve_save_path, metric_name='dice', dataset_name=''):
    distances = dict(dice=0.4, precision=0.3, recall=0.2, jaccard=0.3, topo_2d_err=0.2, pearsonsr=0.4, hausdorff_95=20)
    x, y = read_data(x_path, metric_name), read_data(y_path, metric_name)
    x_test, y_test = read_data(x_test_path, metric_name), read_data(y_test_path, metric_name)
    ext_distance = distances[metric_name]
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    metric_save_path = os.path.join(os.path.dirname(curve_save_path), f'{metric_name}.csv')
    calculate_and_save_metric(polynomial, x_test, y_test, metric_save_path, dataset_name)
    x_fit = np.linspace(max(min(x) - ext_distance, 0), min(max(x) + ext_distance, 1), 1000)
    if metric_name == 'hausdorff_95':
        x_fit = np.linspace(max(min(x) - ext_distance, -200), min(max(x) + ext_distance, 200), 1000)
    y_fit = polynomial(x_fit)
    if metric_name != 'hausdorff_95':
        x_fit = [item for i, item in enumerate(x_fit) if y_fit[i] < 1]
        y_fit = [item for item in y_fit if item < 1]

    # x_ticks = axis_inter(x_fit, ext_distance)
    # y_ticks = axis_inter(y_fit, ext_distance)
    # plt.xticks(x_ticks)
    # plt.yticks(y_ticks)
    plt.scatter(x, y, label='Test set')
    plt.scatter(x_test, y_test, label='Extra test set', color='orange')
    plt.plot(x_fit, y_fit, label='Fitting curve', color='red')
    plt.xlabel(f'Pseudo {metric_name}')
    plt.ylabel(f'Real {metric_name}')
    plt.legend(loc='upper left')
    plt.savefig(curve_save_path)
    plt.close()


def calculate_and_save_metric(polynomial, x, y, save_path, dataset_name):
    metric = dict()
    y_fit = polynomial(x)
    metric['mae'] = mae(y, y_fit)
    metric['correlation'] = pearson_correlation(y, y_fit)
    head = ['dataset_name'] + sorted(metric.keys())
    is_first_write = not os.path.exists(save_path)
    with open(save_path, 'a', newline='') as fp:
        writer = csv.writer(fp)
        if is_first_write:
            writer.writerow(head)
        row = [metric[name] for name in head[1:]]
        row = [dataset_name] + row
        writer.writerow(row)

    return metric


def mae(y, y_fit):
    diff = abs(y_fit-y)
    mean_mae = sum(diff)/len(diff)
    return mean_mae


def pearson_correlation(y, y_fit):
    y = np.array(y)
    y_fit = np.array(y_fit)
    return np.corrcoef(y, y_fit)[0, 1]


def calculate_fitting_function(train_config, universge_config, path_dict=None):
    if path_dict is None:
        csv_base_path = universge_config.spe_config['result_save_dir']
        curve_root_dir = universge_config.spe_config['curve_save_dir']
    else:
        csv_base_path = path_dict['csv_base_path']
        curve_root_dir = path_dict['curve_root_dir']
    dataset_names = [train_config.dataset_name]
    metric_names = ['dice', 'jaccard', 'precision', 'recall', 'pearsonsr', 'hausdorff_95']

    for dataset_name in tqdm(dataset_names):
        for metric_name in metric_names:
            curve_save_path = os.path.join(curve_root_dir, metric_name, f'{dataset_name}_{metric_name}.png')
            if os.path.exists(os.path.dirname(curve_save_path)):
                shutil.rmtree(os.path.dirname(curve_save_path))
            os.makedirs(os.path.dirname(curve_save_path), exist_ok=True)
            draw(f'{csv_base_path}/{dataset_name}/{dataset_name}_test_predict_bootstrap.csv',
                 f'{csv_base_path}/{dataset_name}/{dataset_name}_test_ground_bootstrap.csv',
                 f'{csv_base_path}/{dataset_name}/{dataset_name}_extra_test_predict_bootstrap.csv',
                 f'{csv_base_path}/{dataset_name}/{dataset_name}_extra_test_ground_bootstrap.csv',
                 curve_save_path, metric_name=metric_name, dataset_name=dataset_name)


if __name__ == '__main__':
    from configs.unet_config import UnetConfig
    from configs import universge_config
    train_config = UnetConfig()

    path_dict = {
        'csv_base_path': os.path.join(r'../', universge_config.spe_config['result_save_dir']),
        'curve_root_dir': os.path.join(r'../', universge_config.spe_config['curve_save_dir']),
    }
    calculate_fitting_function(train_config, universge_config, path_dict=path_dict)
