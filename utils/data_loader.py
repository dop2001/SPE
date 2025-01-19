from torch.utils.data import Dataset
import torch
import os
from glob import glob
import cv2


class MedicalDataset(Dataset):
    def __init__(self, img_path, lab_path, mode='train', ratio=0.8):
        img_dir_list, lab_dir_list = (sorted([item for item in os.listdir(img_path) if not item.endswith('.nii.gz')]),
                                      sorted([item for item in os.listdir(lab_path) if not item.endswith('.nii.gz')]))
        assert len(img_dir_list) == len(lab_dir_list)
        self.data, split = [], int(len(img_dir_list) * ratio)

        for idx, (img_dir, lab_dir) in enumerate(zip(img_dir_list, lab_dir_list)):
            img_list = sorted(glob(os.path.join(img_path, img_dir, '*.png')))
            lab_list = sorted(glob(os.path.join(lab_path, lab_dir, '*.png')))
            assert len(img_list) == len(lab_list)
            if mode == 'train' and (idx+1) <= split:
                self.data.extend(zip(img_list, lab_list))
            elif mode == 'valid' and (idx+1) > split:
                self.data.extend(zip(img_list, lab_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (img, lab) = self.data[idx]
        img, lab = cv2.imread(img), cv2.imread(lab)
        gray_image = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        _, single_channel_mask = cv2.threshold(gray_image, 150, 1, cv2.THRESH_BINARY)
        img, lab = torch.tensor(img, dtype=torch.float), torch.tensor(single_channel_mask, dtype=torch.float)
        return img.permute(2, 0, 1), lab.unsqueeze(0)


if __name__ == '__main__':
    img_path = r'../data/train/imagesTr_slice'
    lab_path = r'../data/train/labelsTr_slice'
    dataset = MedicalDataset(img_path, lab_path, mode='valid', ratio=0.8)
    img, lab = dataset[0]
    print(img.shape, lab.shape)
