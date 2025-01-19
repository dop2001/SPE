import skimage.measure
import numpy as np
import os
import glob
# from utils.basic import path2np
from utils.basic import path2np
import SimpleITK as sitk
import csv
from collections import defaultdict
# import utils
smooth = 0.001
# TODO: Multi-region evaluation

def calculate_connect_component(data_np):
    # calculate the number of connect components in the MRI data
    _, num_connect_component = skimage.measure.label(data_np, return_num=True)
    return num_connect_component


def evaluate_connected_component_2d(predicted_image, label_image):
    # get the number of slices in the predicted image
    num_slice = predicted_image.shape[2]
    error_2d_list = []
    # for each slice in the predicted image
    for i in range(num_slice):
        # get the current slice from the predicted image and the label image
        pred_slice = predicted_image[:, :, i]
        label_slice = label_image[:, :, i]
        # if there is any non-zero pixel in the current slice
        if np.sum(label_slice) != 0 or np.sum(pred_slice) != 0:
            # calculate the number of connect components in the current slice of the predicted image and the label image
            pred_cc = calculate_connect_component(pred_slice)
            label_cc = calculate_connect_component(label_slice)
            # calculate the absolute difference of the number of connect components between the predicted image and the label image
            error_2d = abs(label_cc - pred_cc)
            # append the error to the list
            error_2d_list.append(error_2d)
    # convert the list of errors to a numpy array
    error_2d_np = np.array(error_2d_list)
    # return the mean of the errors
    return np.mean(error_2d_np)


def evaluate_connected_component_3d(predicted_image, label_image):
    # calculate the number of connect components in the predicted image and the label image
    pred_cc = calculate_connect_component(predicted_image)
    label_cc = calculate_connect_component(label_image)
    # return the absolute difference of the number of connect components between the predicted image and the label image
    return abs(label_cc - pred_cc)


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_score_multi_region(y_true, y_pred, num_regions):
    dice_scores = []
    for region in range(num_regions):
        # Create binary masks for the current region
        y_true_bin = (y_true == region)
        y_pred_bin = (y_pred == region)
        # Calculate Dice score for the current region
        dice = dice_score(y_true_bin, y_pred_bin)
        dice_scores.append(dice)
    # Average Dice scores over all regions
    return np.mean(dice_scores)

def jaccard_index(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(y_true, y_pred):
    TP = np.sum(y_true * y_pred)  # True Positives
    FP = np.sum((1 - y_true) * y_pred)  # False Positives
    return TP / (TP + FP + smooth)


def recall_score(y_true, y_pred):
    TP = np.sum(y_true * y_pred)  # True Positives
    FN = np.sum(y_true * (1 - y_pred))  # False Negatives
    return TP / (TP + FN + smooth)


def false_positive_rate(y_true, y_pred):
    FP = np.sum((1 - y_true) * y_pred)  # False Positives
    TN = np.sum((1 - y_true) * (1 - y_pred))  # True Negatives
    return FP / (FP + TN + smooth)


def false_negative_rate(y_true, y_pred):
    FN = np.sum(y_true * (1 - y_pred))  # False Negatives
    TP = np.sum(y_true * y_pred)  # True Positives
    return FN / (FN + TP + smooth)


def volume_similarity(y_true, y_pred):
    return 2 * (np.sum(y_pred) - np.sum(y_true)) / (np.sum(y_pred) + np.sum(y_true) + smooth)


def hausdorff_distance(y_true, y_pred):
    label_true = sitk.GetImageFromArray(y_true, isVector=False)
    label_pred = sitk.GetImageFromArray(y_pred, isVector=False)
    hausdorffComputer = sitk.HausdorffDistanceImageFilter()
    hausdorffComputer.Execute(label_true > 0.5, label_pred > 0.5)
    return hausdorffComputer.GetHausdorffDistance()




def mean_value_error(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    result = (y_pred - y_true).sum() / y_true.sum()
    return result


def mean_abs_value_error(y_true, y_pred):
    result = np.abs(mean_value_error(y_true, y_pred))
    return result



def pearsonsr(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    corr_matrix = np.corrcoef(y_true, y_pred)
    return corr_matrix[0, 1]

def hausdorff_distance_95_v1(y_true, y_pred):
    # Convert arrays to SimpleITK images
    label_pred = sitk.GetImageFromArray(y_pred, isVector=False)
    label_true = sitk.GetImageFromArray(y_true, isVector=False)

    # Generate signed Maurer distance maps
    signed_distance_map_true = sitk.SignedMaurerDistanceMap(label_true > 0.5, squaredDistance=False,
                                                            useImageSpacing=True)
    ref_distance_map = sitk.Abs(signed_distance_map_true)

    signed_distance_map_pred = sitk.SignedMaurerDistanceMap(label_pred > 0.5, squaredDistance=False,
                                                            useImageSpacing=True)
    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    # Generate label contours
    ref_surface = sitk.LabelContour(label_true > 0.5, fullyConnected=True)
    seg_surface = sitk.LabelContour(label_pred > 0.5, fullyConnected=True)

    # Calculate distances from one surface to the other
    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)

    # Get all non-zero distances (those are the distances from one surface to the other)
    seg2ref_distances = seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0]
    ref2seg_distances = ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0]

    # Calculate the union of both distance sets
    all_surface_distances = np.concatenate([seg2ref_distances, ref2seg_distances])

    # Calculate HD95
    hd95 = np.percentile(all_surface_distances, 95)

    return hd95
def hausdorff_distance_95_v2(y_true, y_pred):
    # Convert arrays to SimpleITK images
    label_pred = sitk.GetImageFromArray(y_pred, isVector=False)
    label_true = sitk.GetImageFromArray(y_true, isVector=False)

    # Generate signed Maurer distance maps
    signed_distance_map_true = sitk.SignedMaurerDistanceMap(label_true > 0.5, squaredDistance=False, useImageSpacing=True)
    signed_distance_map_pred = sitk.SignedMaurerDistanceMap(label_pred > 0.5, squaredDistance=False, useImageSpacing=True)

    # Generate absolute distance maps
    ref_distance_map = sitk.Abs(signed_distance_map_true)
    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    # Generate label contours
    ref_surface = sitk.LabelContour(label_true > 0.5, fullyConnected=True)
    seg_surface = sitk.LabelContour(label_pred > 0.5, fullyConnected=True)

    # Calculate distances from one surface to the other
    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    # Calculate the number of surface pixels
    num_ref_surface_pixels = int(sitk.GetArrayViewFromImage(ref_surface).sum())
    num_seg_surface_pixels = int(sitk.GetArrayViewFromImage(seg_surface).sum())

    # Convert SimpleITK images to numpy arrays
    seg2ref_distance_map_arr = sitk.GetArrayFromImage(seg2ref_distance_map)
    ref2seg_distance_map_arr = sitk.GetArrayFromImage(ref2seg_distance_map)

    # Get all non-zero distances (those are the distances from one surface to the other)
    seg2ref_distances = seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0]
    ref2seg_distances = ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0]

    # Add zeros to the list of distances until its length is equal to the number of pixels in the contour
    seg2ref_distances = np.concatenate([seg2ref_distances, np.zeros(num_seg_surface_pixels - len(seg2ref_distances))])
    ref2seg_distances = np.concatenate([ref2seg_distances, np.zeros(num_ref_surface_pixels - len(ref2seg_distances))])

    # Calculate the union of both distance sets
    all_surface_distances = np.concatenate([seg2ref_distances, ref2seg_distances])

    # Calculate HD95
    hd95 = np.percentile(all_surface_distances, 95)

    return hd95


def bootstrap_ci(data, statistic=np.mean, alpha=0.05, num_samples=5000):
    n = len(data)
    rng = np.random.RandomState(47)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    stat = np.sort(statistic(samples, axis=1))
    lower = stat[int(alpha / 2 * num_samples)]
    upper = stat[int((1 - alpha / 2) * num_samples)]
    return lower, upper


def cal_avg_bootstrap_confidence_interval(x, decimals=2):
    x_avg = np.average(x)
    bootstrap_ci_result = bootstrap_ci(x)
    #return x_avg, bootstrap_ci_result[0], bootstrap_ci_result[1]
    print(np.round(x_avg, 2), np.round(bootstrap_ci_result[0], 2), np.round(bootstrap_ci_result[1], 2))
    return np.round(x_avg, decimals), np.round(bootstrap_ci_result[0], decimals), np.round(bootstrap_ci_result[1], decimals)

def evaluation_dicts2csv(filename_list, metric_names, metrics_result_list, csv_file_path):
    metric_names = ['filename'] + metric_names

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metric_names)

        writer.writeheader()
        for filename, metrics_result in zip(filename_list, metrics_result_list):
            metrics_result['filename'] = filename
            writer.writerow(metrics_result)

def average_eval_bci(dicts):
    averaged_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            # Skip non-numerical values
            if isinstance(value, (int, float)):
                averaged_dict[key].append(value)
    # Calculate average and bootstrap confidence interval
    averaged_dict = {key: cal_avg_bootstrap_confidence_interval(values) for key, values in averaged_dict.items()}
    return averaged_dict


def main_eval_connected_components():
    # define the directories where the predicted and label files are located
    pred_dir = "D:/program/MedSeg/inference_checkpoints/FD_UNet3D_early_d-qsm_1_subthalamicNucleus_a-None_l-dice_index-1-100.0-0_t-20230618-2152_BEST_0.0"
    label_dir = "D:/Data/ParkinsonOri/label/original/qsm/subthalamicNucleus"
    # get a list of all files in the prediction directory with .nii.gz extension
    pred_files = glob.glob(os.path.join(pred_dir, '*.nii.gz'))
    error_dict = {}
    # for each file in the prediction directory
    for pred_file in pred_files:
        # get the name of the file
        pred_file_name = os.path.basename(pred_file)
        # create the path to the corresponding label file
        label_file = os.path.join(label_dir, pred_file_name)
        # load the predicted image and the label image
        predicted_image, _ = path2np(pred_file)
        label_image, _ = path2np(label_file)
        # calculate the 2D error and 3D error between the predicted image and the label image
        cc_error_2d = evaluate_connected_component_2d(predicted_image, label_image)
        cc_error_3d = evaluate_connected_component_3d(predicted_image, label_image)
        print(f'Filename:{pred_file_name}: 2d CC: {cc_error_2d}, 3d CC: {cc_error_3d}')
        # store the errors in a dictionary
        error_dict[pred_file_name] = (cc_error_2d, cc_error_3d)
    # print the dictionary of errors
    print(error_dict)

def main_eval_connected_components_no_label():
    # define the directories where the predicted and label files are located
    pred_dir = "/Users/fuguanghui/Downloads/UNet3D_3_d-qsm_1_redNucleus_a-None_l-dice_index-1-7.5-3_t-20230704-0024_BEST"
    pred_files = glob.glob(os.path.join(pred_dir, '*.nii.gz'))
    error_list = []

    # for each file in the prediction directory
    for pred_file in pred_files:
        # get the name of the file
        pred_file_name = os.path.basename(pred_file)
        # create the path to the corresponding label file
        predicted_image, _ = path2np(pred_file)
        # calculate the 2D error and 3D error between the predicted image and the label image
        cc_error_3d = np.abs(calculate_connect_component(predicted_image)-2)
        print(f'Filename:{pred_file_name}: 3d CC: {cc_error_3d}')
        # store the errors in a dictionary
        error_list.append(int(cc_error_3d))
    # print the dictionary of errors
    print('Average 3d CC error',np.mean(np.array(error_list)))

# call the main function
if __name__ == '__main__':
    main_eval_connected_components_no_label()
