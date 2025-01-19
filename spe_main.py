import os
from configs.unet_config import UnetConfig
from configs import universge_config
from train import train_segmentation_model
from inference import inference_segmentation_model
from utils.generate_support_set_candidate import generate_support_set_candidate_csv
from utils import universeg_select_support_set
from utils import universeg_pred
from utils.calculate_metrics import calculate_metric
from utils.bootstrap import calculate_average_metric
from utils.draw_curve import calculate_fitting_function


def train_and_inference(train_config):
    train_segmentation_model(train_config)
    for dataset_type in ['test', 'extra_test']:
        inference_segmentation_model(train_config, dataset_type)
        generate_support_set_candidate_csv(train_config, dataset_type)


def generate_pseudo_metric(train_config, dataset_type, repetition_id=1):
    candidate_csv_path = os.path.join(train_config.candidate_csv_dir, f'{dataset_type}_candidate.csv')
    query_img_dir = train_config.dataset_config['img_root_path']
    repetition_name = f'{train_config.dataset_name}_{dataset_type}_{repetition_id}'
    support_set_dir = os.path.join(universge_config.spe_config['save_root_dir'], train_config.dataset_name,
                                   universge_config.spe_config['support_dir_prefix'],
                                   repetition_name)
    selection_list = ['smallest', 'middle', 'largest', 'kmeans', 'seqRandom', 'kmeans_dinov2']
    selection = selection_list[universge_config.spe_config['selection_idx']]
    threshold = universge_config.spe_config['threshold']
    universeg_pred_save_dir = os.path.join(universge_config.spe_config['save_root_dir'], train_config.dataset_name,
                                           universge_config.spe_config['pred_save_dir'], 'train')

    segmentation_model_inference_dir = os.path.join(train_config.inference_config['inference_dir'], dataset_type)
    original_img_dir = os.path.join(train_config.dataset_root, f'{dataset_type}/imagesTs_slice')
    for epoch_id in os.listdir(segmentation_model_inference_dir):
        # 1. Create support set
        segmentation_model_prediction_dir = os.path.join(segmentation_model_inference_dir, epoch_id)
        universeg_select_support_set.main(candidate_csv_path, original_img_dir, segmentation_model_prediction_dir,
                                          f'{support_set_dir}/{epoch_id}', selection)
        # 2. Inference the UniverSeg model
        pred_save_path = f'{universeg_pred_save_dir}_support_set_is_{dataset_type}/{repetition_name}/{epoch_id}'
        universeg_pred.main(query_img_dir, f'{support_set_dir}/{epoch_id}_{selection}',
                            pred_save_path, threshold,
                            depth=train_config.depth_map[train_config.dataset_name])
        calculate_metric(pred_save_path, os.path.join(train_config.dataset_root,
                                                      r'train/labelsTr_slice'))


def pseudo_metric(train_config):
    repetition_num = universge_config.spe_config['repetition_num']
    for idx in range(repetition_num):
        generate_pseudo_metric(train_config, dataset_type='test', repetition_id=idx+1)
    generate_pseudo_metric(train_config, dataset_type='extra_test')


def fitting_function_and_visualization(train_config):
    calculate_average_metric(train_config, universge_config)
    calculate_fitting_function(train_config, universge_config)


def build_spe(train_config):
    # train_and_inference(train_config)
    # pseudo_metric(train_config)
    fitting_function_and_visualization(train_config)


def main():
    # Build SPE framework for every dataset
    dataset_root_dir = r'./datasets'
    for dataset_name in os.listdir(dataset_root_dir):
        print(f'Build SPE on dataset {dataset_name}')
        train_config = UnetConfig(dataset_name=dataset_name)
        build_spe(train_config)


if __name__ == '__main__':
    main()

