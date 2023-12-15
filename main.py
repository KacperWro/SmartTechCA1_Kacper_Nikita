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


#print(cifar10_unwanted_labels)
#print(cifar10_label_names)
#print(cifar10_data_batch_1.keys())

cifar100_test = unpickle("cifar-100-python/test")
cifar100_train = unpickle("cifar-100-python/train")
cifar100_meta = unpickle("cifar-100-python/meta")

print(cifar100_test.keys())
#print(cifar100_meta.keys())
#print(cifar100_meta[b'fine_label_names'])
#print(cifar100_test[b'fine_labels'])
#print(cifar100_test[b'coarse_labels'])


def merge_dicts(dict_list):
    merged_labels = [label for d in dict_list for label in d[b'labels']]
    merged_data = np.concatenate([d[b'data'] for d in dict_list], axis=0)
    return {b'data': merged_data, b'labels': merged_labels}


def replace_numerical_labels_with_names(dict_list):
    for i in range(len(dict_list[b'labels'])):
        numerical_label = dict_list[b'labels'][i]
        dict_list[b'labels'][i] = cifar10_label_names[b'label_names'][numerical_label]
    return dict_list

def replace_numerical_labels_with_fine_names(dict_list):
    for i in range(len(dict_list[b'fine_labels'])):
        numerical_label = dict_list[b'fine_labels'][i]
        dict_list[b'fine_labels'][i] = cifar100_meta[b'fine_label_names'][numerical_label]
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
        print("ERROR: Length of labels is not the same as length of data")
    dict_list[b'data'] = np.array(dict_list[b'data'], dtype=np.uint8)
    return dict_list

cifar10_dict_list = [cifar10_data_batch_1, cifar10_data_batch_2, cifar10_data_batch_3, cifar10_data_batch_4, cifar10_data_batch_5]
cifar10_merged_dict = merge_dicts(cifar10_dict_list)

#print(len(cifar10_merged_dict[b'labels']))
#print(len(cifar10_merged_dict[b'data']))
#print(len(cifar10_merged_dict[b'data'][1]))

img = cifar10_merged_dict[b'data'][100]
reshaped_image = np.reshape(img, (32, 32, 3), order='F')
plt.imshow(reshaped_image)
plt.show()

#print(cifar10_label_names[b'label_names'][1])

cifar10_merged_dict = replace_numerical_labels_with_names(cifar10_merged_dict)
#print(cifar10_merged_dict[b'labels'][1])
#print(type(cifar10_merged_dict[b'labels'][1]))

cifar100_train = replace_numerical_labels_with_fine_names(cifar100_train)
print(cifar100_train[b'fine_labels'])

#print(cifar10_merged_dict.keys())

#print(len(cifar10_merged_dict[b'labels']))
#print(len(cifar10_merged_dict[b'data']))
# print(cifar10_label_names[b'label_names'][0] in cifar10_merged_dict[b'data'])     This line causes a futurewarning
cifar10_merged_dict = remove_unwanted_labels(cifar10_merged_dict)
#print(len(cifar10_merged_dict[b'labels']))
#print(len(cifar10_merged_dict[b'data']))




