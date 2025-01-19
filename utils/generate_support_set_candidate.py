import os
import csv


def write_split_csv(candidate_list, csv_save_path):
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    with open(csv_save_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['File'])
        for entity in candidate_list:
            writer.writerow([entity])


def generate_support_set_candidate_csv(train_config, dataset_type='test'):
    dataset_labels_dir = os.path.join(train_config.dataset_root, f'{dataset_type}/labelsTs_slice')
    candidate_list = sorted([item for item in os.listdir(dataset_labels_dir) if not item.endswith('.nii.gz')])
    csv_save_path = os.path.join(train_config.candidate_csv_dir, f'{dataset_type}_candidate.csv')
    write_split_csv(candidate_list, csv_save_path)


if __name__ == '__main__':
    from configs.unet_config import UnetConfig
    train_config = UnetConfig()
    dataset_type = 'test'
    generate_support_set_candidate_csv(train_config, dataset_type)
