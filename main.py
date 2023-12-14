import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_batch_1 = unpickle("cifar-10-batches-py/data_batch_1")
data_batch_2 = unpickle("cifar-10-batches-py/data_batch_1")
data_batch_3 = unpickle("cifar-10-batches-py/data_batch_1")
data_batch_4 = unpickle("cifar-10-batches-py/data_batch_1")
data_batch_5 = unpickle("cifar-10-batches-py/data_batch_1")

print(data_batch_1.keys())


def merge_dicts(dict_list):
    merged_batch_labels = [label for d in dict_list for label in d[b'batch_label']]
    merged_labels = [label for d in dict_list for label in d[b'labels']]
    merged_data = np.concatenate([d[b'data'] for d in dict_list], axis=0)
    merged_filenames = [label for d in dict_list for label in d[b'filenames']]
    return {b'batch_label': merged_batch_labels, b'data': merged_data, b'labels': merged_labels, b'filenames':merged_filenames}


dict_list = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]
merged_dict = merge_dicts(dict_list)

print(len(merged_dict[b'labels']))
print(len(merged_dict[b'data']))
print(len(merged_dict[b'data'][1]))
