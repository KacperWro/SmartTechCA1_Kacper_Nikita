import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
cifar10_unwanted_labels = [cifar10_label_names[b'label_names'][0],
                           cifar10_label_names[b'label_names'][6],
                           cifar10_label_names[b'label_names'][8]]


cifar100_test = unpickle("cifar-100-python/test")
cifar100_train = unpickle("cifar-100-python/train")
cifar100_meta = unpickle("cifar-100-python/meta")

cifar100_wanted_fine_labels = [
    cifar100_meta[b'fine_label_names'][19],# cattle
    cifar100_meta[b'fine_label_names'][34],# fox
    cifar100_meta[b'fine_label_names'][41],# lawnmower
    cifar100_meta[b'fine_label_names'][65],# rabbit
    cifar100_meta[b'fine_label_names'][80],# squirrel
    cifar100_meta[b'fine_label_names'][89]# tractor
]

cifar100_wanted_coarse_labels = [
    cifar100_meta[b'coarse_label_names'][14], # baby, boy, girl, man, woman
    cifar100_meta[b'coarse_label_names'][17], # maple, oak, palm, pine, willow
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
    merged_label_names = [label for d in labels_dict_list for label in d.get(b'label_names', d.get(b'fine_label_names', []))]
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

    if len(dict_list[b'fine_labels']) == len(dict_list[b'data']) and len(dict_list[b'fine_labels']) == len(dict_list[b'coarse_labels']):
        counter = 0
        while counter < len(dict_list[b'data']):
            if dict_list[b'coarse_labels'][counter] not in cifar100_wanted_coarse_labels and dict_list[b'fine_labels'][counter] not in cifar100_wanted_fine_labels:
                del dict_list[b'fine_labels'][counter]
                del dict_list[b'coarse_labels'][counter]
                del dict_list[b'data'][counter]
            else:
                counter = counter + 1
    else:
        raise Exception("ERROR: Length of labels is not the same as length of data")
    dict_list[b'data'] = np.array(dict_list[b'data'], dtype=np.uint8)
    return dict_list


def count_labels(dict_list):
    for i in range(len(combined_labels_dict[b'label_names'])):
        counter = 0

        for j in range(len(dict_list[b'labels'])):
            if dict_list[b'labels'][j] == combined_labels_dict[b'label_names'][i]:
                counter = counter + 1

        print(combined_labels_dict[b'label_names'][i], " count: ", counter)


combined_labels_list = [cifar10_label_names, cifar100_meta]
combined_labels_dict = merge_label_names(combined_labels_list)

cifar10_dict_list = [cifar10_data_batch_1, cifar10_data_batch_2, cifar10_data_batch_3, cifar10_data_batch_4, cifar10_data_batch_5]
cifar10_merged_dict = merge_dicts(cifar10_dict_list)
cifar10_merged_dict = replace_numerical_labels_with_names(cifar10_merged_dict)
cifar10_merged_dict = remove_unwanted_labels(cifar10_merged_dict)

cifar100_train = replace_numerical_labels_with_fine_names(cifar100_train)
cifar100_train = remove_unwanted_labels_cifar_100(cifar100_train)

entire_dict_list = [cifar10_merged_dict, cifar100_train]
merged_data_dict = merge_dicts_with_different_labels(entire_dict_list)

print("\nLabels length", len(merged_data_dict[b'labels']))
print("Data rows length", len(merged_data_dict[b'data']))
print("Data columns length", len(merged_data_dict[b'data'][1]))

img = merged_data_dict[b'data'][0]
reshaped_image = np.reshape(img, (32, 32, 3), order='F')
plt.imshow(reshaped_image)
plt.show()

print(len(cifar100_train[b'coarse_labels']))
print(len(cifar100_train[b'fine_labels']))
print(len(cifar100_train[b'data']))

print("\nLength of labels after removing unwanted labels:", len(merged_data_dict[b'labels']))
print("Length of data after removing unwanted labels:", len(merged_data_dict[b'data']))

print("\nCOUNTS PER LABEL AFTER REMOVING UNWANTED LABELS")
count_labels(merged_data_dict)



