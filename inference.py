import os

import torch
from models.UNet import UNet
import cv2
from tqdm import tqdm
from utils.png2nii import png_to_nifti

from glob import glob
import re
import shutil
from utils.calculate_metrics import calculate_metric


def inference(img_root_path, save_root_path, weight_path, depth=256):
    os.makedirs(save_root_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    dir_name_list = os.listdir(img_root_path)
    for dir_name in tqdm(dir_name_list):
        img_dir_path = os.path.join(img_root_path, dir_name)
        save_dir_path = os.path.join(save_root_path, dir_name.replace('_0000', ''))
        os.makedirs(save_dir_path, exist_ok=True)
        for img_name in sorted(os.listdir(img_dir_path)):
            img_path = os.path.join(img_dir_path, img_name)
            img = torch.tensor(cv2.imread(img_path), dtype=torch.float).cuda().permute(2, 0, 1).unsqueeze(0)
            output = model(img)
            result = output.reshape(1, 2, 128, 128).argmax(1)[0]
            result = result.cpu().numpy() * 255
            result = result.astype('uint8')
            cv2.imwrite(os.path.join(save_dir_path, img_name), result)
        png_to_nifti(save_root_path, dir_name.replace('_0000', ''), depth=depth)


def inference_segmentation_model(train_config, dataset_type='test'):
    dataset_root_dir = train_config.dataset_root
    save_root_dir = train_config.inference_config['inference_dir']
    depth = train_config.depth_map[train_config.dataset_name]
    pth_path = train_config.network_config['pth_save_path']
    pth_path_list = sorted(glob(os.path.join(pth_path, '*.pt')),
                           key=lambda x: float(re.search(r'(\d+\.+\d+)', x).group()))[::5]
    img_root_path = os.path.join(dataset_root_dir, f'{dataset_type}/imagesTs_slice')
    save_root_path = os.path.join(save_root_dir, dataset_type)
    for pth_path in pth_path_list:
        num_id = os.path.basename(pth_path).split('_')[0]
        save_path = os.path.join(save_root_path, num_id)
        inference(img_root_path, save_path, weight_path=pth_path, depth=depth)
        calculate_metric(save_path, os.path.join(dataset_root_dir, f'{dataset_type}/labelsTs_slice'))


if __name__ == '__main__':
    from configs.unet_config import UnetConfig
    train_config = UnetConfig()
    dataset_type = 'test'  # test or extra_test
    inference_segmentation_model(train_config, dataset_type)
