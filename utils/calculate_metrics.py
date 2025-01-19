import os
from tqdm import tqdm
from utils.eval_ci import *
import utils.basic as utils
import csv


def calculate_std(eval_np):
    return np.std(eval_np)


def pred_eval(pred_base_path, label_base_path, evaluation_metrics, evaluation_funcs, csv_file_path):
    assert len(evaluation_metrics) == len(evaluation_funcs)
    pred_path_list = glob.glob(f'{pred_base_path}/*.nii.gz')
    filename_list = []
    evaluation_metrics_result_list = []
    for pred_path in tqdm(pred_path_list):
        filename = os.path.basename(pred_path)
        label_path = f'{label_base_path}/{filename}'
        pred_np, _ = utils.path2np(pred_path)
        label_np, _ = utils.path2np(label_path)
        try:
            evaluation_matrix_result = {metric: func(label_np, pred_np) for metric, func in
                                        zip(evaluation_metrics, evaluation_funcs)}
            evaluation_matrix_result_rounded = {key: round(value, 2) for key, value in
                                                evaluation_matrix_result.items()}
        except Exception as e:
            print('[ERROR]', filename, e)
        filename_list.append(filename)
        evaluation_metrics_result_list.append(evaluation_matrix_result)
    evaluation_dicts2csv(filename_list, evaluation_metrics, evaluation_metrics_result_list, csv_file_path)
    evaluation_metrics_result_avg = average_eval_bci(evaluation_metrics_result_list)
    print('Average:', evaluation_metrics_result_avg)


def calculate_metric(pred_base_path, label_base_path):
    evaluation_metrics = ['dice', 'hausdorff_95', 'topo_2d_err',
                          'topo_3d_err', 'jaccard', 'precision',
                          'recall', 'mver', 'maver', 'pearsonsr']
    evaluation_funcs = [dice_score, hausdorff_distance_95_v2, evaluate_connected_component_2d,
                        evaluate_connected_component_3d, jaccard_index, precision_score,
                        recall_score, mean_value_error, mean_abs_value_error, pearsonsr]
    csv_file_path = f'{pred_base_path}/eval.csv'
    print(pred_base_path, label_base_path)
    pred_eval(pred_base_path, label_base_path, evaluation_metrics, evaluation_funcs, csv_file_path)


def calculate_metrics(pred_base_path, label_base_path):
    dir_name_list = os.listdir(pred_base_path)
    for dir_name in sorted(dir_name_list):
        pred_path = os.path.join(pred_base_path, dir_name, 'prediction')
        calculate_metric(pred_path, label_base_path)


if __name__ == '__main__':
    from configs.unet_config import UnetConfig
    train_config = UnetConfig()
    pred_base_path = os.path.join('../', train_config.inference_config['inference_dir'], 'test')
    label_base_path = os.path.join('../', train_config.dataset_root, 'test/labelsTs_slice')
    calculate_metrics(pred_base_path, label_base_path)

