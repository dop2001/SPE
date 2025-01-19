import os
import csv
import shutil
from utils.eval_ci import *
import numpy as np
import matplotlib.pyplot as plt


def read_cvs(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        column_data = {column: [] for column in reader.fieldnames}
        for row in reader:
            for column in reader.fieldnames:
                column_data[column].append(row[column])
    return reader.fieldnames, column_data


def write_cvs(file_path, avg_result):
    head_list = ['name', 'dice', 'hausdorff_95', 'topo_2d_err', 'topo_3d_err',
                 'jaccard', 'precision', 'recall', 'pearsonsr']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(head_list)
        for item in avg_result:
            datas = [item['num']]
            for k in head_list[1:]:
                k = f'{k}_avg'
                datas.append(f'{np.round(item[k][0], 5)}[{np.round(item[k][1], 5)},'
                             f'{np.round(item[k][2], 5)}]')
            writer.writerow(datas)


def batch_csv(infer_dir, infer_type):
    if infer_type == 'unet_infer_dir':
        infer_dir_list = [infer_dir]
    elif infer_type == 'useg_infer_dir':
        infer_dir_list = sorted([os.path.join(infer_dir, dir_name) for dir_name in os.listdir(infer_dir)])
    else:
        raise ValueError(f'{infer_type} is not support!')
    field_names, raw_result, epoch_num_list = None, [], []
    for infer_dir in infer_dir_list:
        epoch_num_list = sorted(os.listdir(infer_dir))
        data_dict = dict()
        for epoch_num in epoch_num_list:
            csv_path = os.path.join(infer_dir, epoch_num, 'eval.csv')
            field_names, data_dict[epoch_num] = read_cvs(csv_path)
        raw_result.append(data_dict)
    return epoch_num_list, field_names, raw_result


def calculate_average_metric_for_signal_fold(infer_dir, infer_type):
    num_list, fieldnames, raw_result = batch_csv(infer_dir, infer_type)
    avg_result = []
    for num in num_list:
        temp = dict(filename=raw_result[0]['0']['filename'], num=num)
        for file_name in fieldnames[1:]:
            column_data = np.asarray([x[num][file_name] for x in raw_result]).astype(float)
            avg_data = list(np.mean(column_data, axis=0))
            temp[file_name] = avg_data
            avg_data = [x for x in avg_data if x < 10000]   # Avoid outlier interference
            temp[f'{file_name}_avg'] = cal_avg_bootstrap_confidence_interval(avg_data, decimals=5)
        avg_result.append(temp)
    return avg_result


def calculate_average_metric(train_config, universge_config, infer_dir_map=None):
    dataset_name = train_config.dataset_name
    dataset_types = ['test', 'extra_test']
    result_save_dir = os.path.join(r'../', universge_config.spe_config['result_save_dir'],
                                   train_config.dataset_name)
    if infer_dir_map is None:
        infer_dir_map = {
            'unet_infer_dir': os.path.join(train_config.save_root_path, dataset_name,
                                           train_config.inference_dir_suffix),
            'useg_infer_dir': os.path.join(train_config.save_root_path, dataset_name,
                                           universge_config.spe_config['pred_save_dir']),
        }
        result_save_dir = os.path.join(universge_config.spe_config['result_save_dir'],
                                            train_config.dataset_name)
    os.makedirs(result_save_dir, exist_ok=True)
    for (inter_type, infer_root_dir) in infer_dir_map.items():
        for dataset_type in dataset_types:
            metric_type = 'ground'
            infer_dir = os.path.join(infer_root_dir, dataset_type)
            if inter_type == 'useg_infer_dir':
                metric_type = 'predict'
                infer_dir = os.path.join(infer_root_dir, f'train_support_set_is_{dataset_type}')
            avg_result = calculate_average_metric_for_signal_fold(infer_dir, infer_type=inter_type)
            csv_file_name = f'{dataset_name}_{dataset_type}_{metric_type}_bootstrap.csv'
            result_save_path = os.path.join(result_save_dir, csv_file_name)
            if os.path.exists(result_save_path):
                os.remove(result_save_path)
            write_cvs(result_save_path, avg_result)


if __name__ == '__main__':
    from configs.unet_config import UnetConfig
    from configs import universge_config
    train_config = UnetConfig()
    infer_dir_map = {
        'unet_infer_dir': os.path.join(r'../process', train_config.dataset_name,
                                       train_config.inference_dir_suffix),
        'useg_infer_dir': os.path.join(r'../process', train_config.dataset_name,
                                       universge_config.spe_config['pred_save_dir']),
    }
    calculate_average_metric(train_config, universge_config, infer_dir_map=infer_dir_map)


