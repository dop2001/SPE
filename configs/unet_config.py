from datetime import datetime
import os
import torch


class UnetConfig:
    def __init__(self, dataset_name='JSRT'):
        self.dataset_name = dataset_name
        self.save_root_path = r'process'
        self.log_dir_suffix = r'segmentation_model/logs'
        self.run_dir_suffix = r'segmentation_model/runs'
        self.pth_dir_suffix = r'segmentation_model/pths'
        self.inference_dir_suffix = r'segmentation_model/inference'
        self.candidate_csv_suffix = r'universeg_model/candidate_csv'

        self.candidate_csv_dir = os.path.join(self.save_root_path, self.dataset_name, self.candidate_csv_suffix)

        # log file
        self.loger_config = {
            'log_path': os.path.join(self.save_root_path, self.dataset_name, self.log_dir_suffix,
                                     r'{:%Y_%m_%d-%H_%M_%S}.log'.format(datetime.now())),
        }

        # tensorboard file
        self.summaryWriter_config = {
            'summary_path': os.path.join(self.save_root_path, self.dataset_name, self.run_dir_suffix,
                                         r'{:%Y_%m_%d-%H_%M_%S}'.format(datetime.now())),
        }

        # dataset path
        self.dataset_root = r'./datasets/{}'.format(self.dataset_name)
        self.dataset_config = {
            'img_root_path': os.path.join(self.dataset_root, 'train/imagesTr_slice'),
            'lab_root_path': os.path.join(self.dataset_root, 'train/labelsTr_slice')
        }

        # network
        self.network_config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'image_size': (128, 128),
            'epoch': 100,
            'batch_size': 8,
            'lr': 1e-4,
            'num_workers': 0,
            'iterate_step': 300,
            'pth_save_path': os.path.join(self.save_root_path, dataset_name, self.pth_dir_suffix),
        }

        self.inference_config = {
            'inference_dir': os.path.join(self.save_root_path, dataset_name, self.inference_dir_suffix)
        }

        self.depth_map = {
            'JSRT': 1,
            'PSFHS': 1,
            'SCD': 1,
            'ISIC2018': 1,
            'HC18': 1,
            '3D-IRCADB': 256,
        }
