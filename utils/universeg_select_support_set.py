import numpy as np
import pandas as pd
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import random
from glob import glob


def select_slice(subject_names, image_dir, label_dir, support_set_dir, selection='largest', min_area=10):
    support_images_path = os.path.join(f'{support_set_dir}_{selection}', 'images')
    support_labels_path = os.path.join(f'{support_set_dir}_{selection}', 'labels')

    # Ensure the support directories exist
    os.makedirs(support_images_path, exist_ok=True)
    os.makedirs(support_labels_path, exist_ok=True)
    img_list = []
    for subject in subject_names:
        temp_list = []
        for file_path in sorted(glob(f'{image_dir}/{subject}/*.png')):
            lab_path = os.path.join(label_dir, subject.replace('_0000', ''), os.path.basename(file_path))
            label_image = plt.imread(lab_path)
            label_image = np.array(label_image, dtype=np.uint8)
            current_sum_or_area = np.sum(label_image) if selection != 'middle' else np.sum(label_image > 0)
            if current_sum_or_area >= min_area:
                temp_list.append(file_path)
        img_list.extend(temp_list)
    if len(img_list) > 64:
         random.shuffle(img_list)
         img_list = img_list[:64]

    for selected_file_path in img_list:
        slice_num = int(os.path.basename(selected_file_path).split('_')[1])
        subject = os.path.basename(os.path.dirname(selected_file_path)).replace('_0000', '')
        label_dest_file_path = f'{support_labels_path}/{subject}_{slice_num}.png'
        image_dest_file_path = f'{support_images_path}/{subject}_{slice_num}.png'

        # Copy label
        copyfile(os.path.join(label_dir, subject, os.path.basename(selected_file_path)), label_dest_file_path)
        # Copy corresponding MRI image
        image_file_path = selected_file_path
        copyfile(image_file_path, image_dest_file_path)


def main(data_split_csv_path, image_dir, label_dir, support_set_base_dir, selection):
    train_subjects_df = pd.read_csv(data_split_csv_path)
    subject_names = train_subjects_df['File'].tolist()
    select_slice(subject_names, image_dir, label_dir, support_set_base_dir, selection=selection, min_area=10)
    print('Processing complete.')
