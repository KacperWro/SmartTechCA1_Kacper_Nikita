import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import statistics
from skimage.color import rgb2gray, gray2rgb

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


cifar10_data_batch_1 = unpickle("cifar-10-batches-py/data_batch_1")
cifar10_data_batch_2 = unpickle("cifar-10-batches-py/data_batch_1")
cifar10_data_batch_3 = unpickle("cifar-10-batches-py/data_batch_1")
cifar10_data_batch_4 = unpickle("cifar-10-batches-py/data_batch_1")
cifar10_data_batch_5 = unpickle("cifar-10-batches-py/data_batch_1")
cifar10_label_names = unpickle("cifar-10-batches-py/batches.meta")
cifar10_test_batch = unpickle("cifar-10-batches-py/test_batch")
cifar10_unwanted_labels = [cifar10_label_names[b'label_names'][0],
                           cifar10_label_names[b'label_names'][6],
                           cifar10_label_names[b'label_names'][8]]

cifar100_test = unpickle("cifar-100-python/test")
cifar100_train = unpickle("cifar-100-python/train")
cifar100_meta = unpickle("cifar-100-python/meta")


cifar100_wanted_fine_labels = [
    cifar100_meta[b'fine_label_names'][19],  # cattle
    cifar100_meta[b'fine_label_names'][34],  # fox
    cifar100_meta[b'fine_label_names'][41],  # lawnmower
    cifar100_meta[b'fine_label_names'][65],  # rabbit
    cifar100_meta[b'fine_label_names'][80],  # squirrel
    cifar100_meta[b'fine_label_names'][89]  # tractor
]

cifar100_wanted_coarse_labels = [
    cifar100_meta[b'coarse_label_names'][14],  # baby, boy, girl, man, woman
    cifar100_meta[b'coarse_label_names'][17],  # maple, oak, palm, pine, willow
    cifar100_meta[b'coarse_label_names'][18]  # bicycle, bus, motorcycle, pickup truck, train
]


def merge_dicts(dict_list):
    merged_labels = [label for d in dict_list for label in d[b'labels']]
    merged_data = np.concatenate([d[b'data'] for d in dict_list], axis=0)
    return {b'data': merged_data, b'labels': merged_labels}


def merge_dicts_with_different_labels(dict_list):
    merged_labels = [label for d in dict_list for label in d.get(b'labels', d.get(b'fine_labels', []))]
    merged_data = np.concatenate([d[b'data'] for d in dict_list], axis=0)
    return {b'data': merged_data, b'labels': merged_labels}


def merge_label_names(labels_dict_list):
    merged_label_names = [label for d in labels_dict_list for label in
                          d.get(b'label_names', d.get(b'fine_label_names', []))]
    return {b'label_names': merged_label_names}


def replace_numerical_labels_with_names(dict_list):
    for i in range(len(dict_list[b'labels'])):
        numerical_label = dict_list[b'labels'][i]
        dict_list[b'labels'][i] = cifar10_label_names[b'label_names'][numerical_label]
    return dict_list


def replace_numerical_labels_with_fine_names(dict_list):
    for i in range(len(dict_list[b'fine_labels'])):
        numerical_fine_label = dict_list[b'fine_labels'][i]
        dict_list[b'fine_labels'][i] = cifar100_meta[b'fine_label_names'][numerical_fine_label]
        numerical_coarse_label = dict_list[b'coarse_labels'][i]
        dict_list[b'coarse_labels'][i] = cifar100_meta[b'coarse_label_names'][numerical_coarse_label]
    return dict_list


def remove_unwanted_labels(dict_list):
    dict_list[b'data'] = dict_list[b'data'].tolist()

    if len(dict_list[b'labels']) == len(dict_list[b'data']):
        counter = 0
        while counter < len(dict_list[b'labels']):
            if dict_list[b'labels'][counter] in cifar10_unwanted_labels:
                del dict_list[b'labels'][counter]
                del dict_list[b'data'][counter]
            else:
                counter = counter + 1
    else:
        raise Exception("ERROR: Length of labels is not the same as length of data")
    dict_list[b'data'] = np.array(dict_list[b'data'], dtype=np.uint8)
    return dict_list


def remove_unwanted_labels_cifar_100(dict_list):
    dict_list[b'data'] = dict_list[b'data'].tolist()

    if len(dict_list[b'fine_labels']) == len(dict_list[b'data']) and len(dict_list[b'fine_labels']) == len(
            dict_list[b'coarse_labels']):
        counter = 0
        while counter < len(dict_list[b'data']):
            if dict_list[b'coarse_labels'][counter] not in cifar100_wanted_coarse_labels \
                    and dict_list[b'fine_labels'][counter] not in cifar100_wanted_fine_labels:

                del dict_list[b'fine_labels'][counter]
                del dict_list[b'coarse_labels'][counter]
                del dict_list[b'data'][counter]
            else:
                counter = counter + 1
    else:
        raise Exception("ERROR: Length of labels is not the same as length of data")
    dict_list[b'data'] = np.array(dict_list[b'data'], dtype=np.uint8)
    return dict_list


def count_labels(data_dict, labels_dict):
    for i in range(len(labels_dict[b'label_names'])):
        counter = 0

        for j in range(len(data_dict[b'labels'])):
            if data_dict[b'labels'][j] == labels_dict[b'label_names'][i]:
                counter = counter + 1

        if counter > 0:
            print(labels_dict[b'label_names'][i], " count: ", counter)


# Displaying a sample image
img = cifar100_train[b'data'][500]
reshaped_image = np.reshape(img, (32, 32, 3), order='F')
plt.imshow(reshaped_image)
plt.show()

# Getting label names from cifar10 and cifar100 and combining them into a single dictionary
combined_labels_list = [cifar10_label_names, cifar100_meta]
combined_labels_dict = merge_label_names(combined_labels_list)

# Replacing numerical values in cifar10 test batch with their corresponding label names + removing any unwanted labels
cifar10_test_batch = replace_numerical_labels_with_names(cifar10_test_batch)
cifar10_test_batch = remove_unwanted_labels(cifar10_test_batch)

# Replacing numerical values in cifar100 test batch with their corresponding label names + removing any unwanted labels
cifar100_test = replace_numerical_labels_with_fine_names(cifar100_test)
cifar100_test = remove_unwanted_labels_cifar_100(cifar100_test)

# Combining cifar10 and cifar100 training data into a single dictionary
all_tests_list = [cifar10_test_batch, cifar100_test]
merged_tests_dict = merge_dicts_with_different_labels(all_tests_list)

# Combining all batches of cifar10 training data into a single dictionary
cifar10_dict_list = [cifar10_data_batch_1, cifar10_data_batch_2, cifar10_data_batch_3,
                     cifar10_data_batch_4, cifar10_data_batch_5]
cifar10_merged_dict = merge_dicts(cifar10_dict_list)

# Replacing numerical values in cifar10 training dict with their corresponding label names + removing any unwanted labels
cifar10_merged_dict = replace_numerical_labels_with_names(cifar10_merged_dict)
cifar10_merged_dict = remove_unwanted_labels(cifar10_merged_dict)

# Replacing numerical values in cifar100 training dict with their corresponding label names + removing any unwanted labels
cifar100_train = replace_numerical_labels_with_fine_names(cifar100_train)
cifar100_train = remove_unwanted_labels_cifar_100(cifar100_train)

# Combining cifar10 and cifar100 training data into a single dictionary
entire_dict_list = [cifar10_merged_dict, cifar100_train]
merged_data_dict = merge_dicts_with_different_labels(entire_dict_list)

print("\nTotal number of training images: ", len(merged_data_dict[b'data']))
print("Total number of training labels: ", len(merged_data_dict[b'labels']))
print("\nTotal number of training images per class")
print("=====================================================")
count_labels(merged_data_dict, combined_labels_dict)
print("=====================================================")

print("\nTotal number of test images: ", len(merged_tests_dict[b'data']))
print("Total number of test labels: ", len(merged_tests_dict[b'labels']))
print("\nTotal number of test images per class")
print("=====================================================")
count_labels(merged_tests_dict, combined_labels_dict)
print("=====================================================")


merged_data_dict[b'data'] = [[float(num) for num in sublist] for sublist in merged_data_dict[b'data']]
def normalise_image_pixel_values(dict_list):
    for i in range(len(dict_list[b'data'])):
        original_array = np.array(dict_list[b'data'][i], dtype = float)
        result_array = original_array / 255.0
        dict_list[b'data'][i] = result_array.tolist()
    return dict_list

def grayscale_images(dict_list):
    for i in range(len(dict_list[b'data'])):
        img = dict_list[b'data'][i]
        reshaped_image = np.reshape(img, (32, 32, 3), order='F') #images are 32 x 32
        grayscale_image = rgb2gray(reshaped_image)
        dict_list[b'data'][i] = grayscale_image
        

merged_data_dict = normalise_image_pixel_values(merged_data_dict)
grayscale_images(merged_data_dict)
print(len(merged_data_dict[b'data'][3]))

# sample image after normalisation and grayscaling
img = merged_data_dict[b'data'][3]
reshaped_image = np.reshape(img, (32, 32, 1), order='F')
plt.imshow(reshaped_image)
plt.show()