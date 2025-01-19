import torch
import nibabel as nib
import os
import numpy as np
import torchio as tio
import torch.backends.cudnn as cudnn
from collections import defaultdict


def reproducibility(args, seed):
    """
    Set the seed for all possible sources of randomness to ensure reproducibility.
    :param args: command line arguments
    :param seed: seed for the random number generators
    """
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def mkdir(data_path):
    """
    Create a new directory if it doesn't already exist.
    :param data_path: path where the directory should be created
    """
    os.makedirs(data_path, exist_ok=True)


def path2nib(data_path):
    """
    Load a NIfTI file.
    :param data_path: path to the NIfTI file
    :return: NIfTI file
    """
    return nib.load(data_path)


def path2np(data_path):
    """
    Load a NIfTI file and return its data as a numpy array and its affine.
    :param data_path: path to the NIfTI file
    :return: data as a numpy array and its affine
    """
    mri = path2nib(data_path)
    mri_np = np.asarray(mri.get_fdata(dtype=np.float32))
    mri_affine = mri.affine
    return mri_np, mri_affine


def path2tensor(data_path):
    """
    Load a NIfTI file and return its data as a torch tensor.
    :param data_path: path to the NIfTI file
    :return: data as a torch tensor
    """
    mri_nii = nib.load(data_path)
    mri_np = mri_nii.get_fdata(dtype=np.float32)
    mri_tensor = torch.as_tensor(mri_np)
    return mri_tensor


def path2tio(data_path):
    """
    Create a TorchIO ScalarImage from a file path.
    :param data_path: path to the file
    :return: TorchIO ScalarImage
    :return: dataname
    """
    dataname = os.path.basename(data_path)
    tio_img = tio.ScalarImage(data_path)
    return dataname, tio_img


def path2tio_label(data_path):
    """
    Create a TorchIO LabelMap from a file path.
    :param data_path: path to the file
    :return: TorchIO LabelMap
    """
    return tio.LabelMap(data_path)




def path2tiosubject(data_path, label_path):
    """
    Create a TorchIO Subject from image and label paths.
    :param data_path: path to the image file
    :param label_path: path to the label file
    :return: TorchIO Subject
    """
    tio_img = tio.ScalarImage(data_path)
    # resampling = tio.Resample(target=tio_img)
    # label = resampling(tio.LabelMap(label_path))
    label = tio.LabelMap(label_path)
    # Cast label to float
    label.data = label.data.float()

    return tio.Subject(
        image=tio_img,
        label=label,
        name=os.path.basename(data_path)
    )

def path2tiosubject_no_label(data_path):
    """
    Create a TorchIO Subject from image and label paths.
    :param data_path: path to the image file
    :param label_path: path to the label file
    :return: TorchIO Subject
    """
    tio_img = tio.ScalarImage(data_path)
    return tio.Subject(
        image=tio_img,
        name=os.path.basename(data_path)
    )

def path2tiosubject_multi(data_paths, label_paths, data_names, label_names):
    """
    Create a TorchIO Subject from multiple image and label paths.
    :param data_paths: list of paths to the image files
    :param label_paths: list of paths to the label files
    :param data_names: list of names for the image files
    :param label_names: list of names for the label files
    :return: TorchIO Subject
    """

    if len(data_paths) != len(data_names):
        raise ValueError("The number of data paths and data names should match.")
    if len(label_paths) != len(label_names):
        raise ValueError("The number of label paths and label names should match.")

    tio_subject_dict = {}

    for i, path in enumerate(data_paths):
        tio_img = tio.ScalarImage(path)
        tio_subject_dict[data_names[i]] = tio_img

    for i, path in enumerate(label_paths):
        tio_label = tio.Resample(target=tio_img)(tio.LabelMap(path))
        tio_label.data = tio_label.data.float()
        tio_subject_dict[label_names[i]] = tio_label

    return tio.Subject(
        **tio_subject_dict,
        name=os.path.basename(data_paths[0])
    )


def save_nii(mri_np, mri_affine, save_path):
    """
    Save a numpy array as a NIfTI file.
    :param mri_np: MRI data as a numpy array
    :param mri_affine: affine of the MRI data
    :param save_path: path where the NIfTI file should be saved
    """
    mri_np = np.array(mri_np, dtype=np.float32)
    mri_nii = nib.Nifti1Image(mri_np, mri_affine)
    nib.save(mri_nii, save_path)


def zip_nii(path):
    """
    Save a NIfTI file as a compressed file.
    :param path: path to the NIfTI file
    """
    mri = path2nib(path)
    nib.save(mri, path.replace('.nii', '.nii.gz'))


def one_hot(data_tensor, num_classes):
    """
    Apply one-hot encoding to a tensor.
    :param data_tensor: tensor to be one-hot encoded
    :param num_classes: number of classes for one-hot encoding
    :return: one-hot encoded tensor
    """
    return tio.transforms.OneHot(num_classes=num_classes)(data_tensor)


def average_dicts(dicts, decimals=4):
    averaged_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            # Skip non-numerical values
            if isinstance(value, (int, float)):
                averaged_dict[key].append(value)
    averaged_dict = {key: np.round(sum(values) / len(values), decimals) for key, values in averaged_dict.items()}
    return averaged_dict


def dict2str(evaluation_metrics, dicts):
    output = ', '.join([f"{key}: {dicts[key]}" for key in evaluation_metrics])
    return output


def clean_ds_store(folder_name_list):
    return [file for file in folder_name_list if file != ".DS_Store"]


def filter_folder_list(folder_name_list, filter_list):
    filtered_list = []
    for file in folder_name_list:
        should_filter = False
        for filter_str in filter_list:
            if file.endswith(filter_str):
                should_filter = True
                break
        if not should_filter:
            filtered_list.append(file)
    return filtered_list
