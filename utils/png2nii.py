import glob
import os
from PIL import Image
import nibabel as nib
import numpy as np


def png_to_nifti(pred_dir, subject_name, depth=256):
    input_dir = os.path.join(pred_dir, subject_name)
    output_file = os.path.join(pred_dir, f"{subject_name}_0000.nii.gz")
    reference_nifti_affine = np.eye(4)
    volume = np.zeros((256, 256, depth), dtype=np.uint8)
    for slice_index in range(depth):
        file_name = f'slice_{slice_index}_image.png'
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            print(file_path)
            current_slice = load_and_resize_image(file_path)
        else:
            current_slice = np.zeros((256, 256), dtype=np.uint8)
        volume[:, :, slice_index] = current_slice
    nifti_img = nib.Nifti1Image(volume, reference_nifti_affine)
    nib.save(nifti_img, output_file)


def load_and_resize_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('L')
    img_resized = img.resize(target_size)
    img_np = np.array(img_resized)
    binarized_img = np.where(img_np > 127, 1, 0)
    return binarized_img


if __name__ == '__main__':
    pre_dir = r'./universeg_data/T1/slice/slice_128_128_filter/extra_test/labelsTs_slice'
    subject_name_list = os.listdir(pre_dir)
    for subject_name in subject_name_list:
        png_to_nifti(pre_dir, subject_name)
