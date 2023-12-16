import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import statistics
from skimage.color import rgb2gray, gray2rgb
from imgaug import augmenters as iaa
import matplotlib.image as mpimg
import random

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


def normalise_image_pixel_values(dict_list):
    for i in range(len(dict_list[b'data'])):
        original_array = np.array(dict_list[b'data'][i], dtype = float)
        result_array = original_array / 255.0
        dict_list[b'data'][i] = result_array.tolist()
    return dict_list


def grayscale_images(dict_list):
    for i in range(len(dict_list[b'data'])):
        img = dict_list[b'data'][i]
        reshaped_image = np.reshape(img, (32, 32, 3), order='F')
        grayscale_image = cv2.cvtColor(reshaped_image, cv2.COLOR_RGB2GRAY)
        flattened_grayscale = np.tile(grayscale_image.flatten(), 3)
        dict_list[b'data'][i] = flattened_grayscale
        

def equalize_images(dict_list):
    for i in range(len(dict_list[b'data'])):
        equalized_image = cv2.equalizeHist(dict_list[b'data'][i])
        equalized_image_flattened = equalized_image.ravel()
        dict_list[b'data'][i] = equalized_image_flattened
    return dict_list


grayscale_images(merged_data_dict)
merged_data_dict = equalize_images(merged_data_dict)
merged_data_dict[b'data'] = [[float(num) for num in sublist] for sublist in merged_data_dict[b'data']]
merged_data_dict = normalise_image_pixel_values(merged_data_dict)
print(len(merged_data_dict[b'data'][3]))

# sample training image after normalisation and grayscaling
img = merged_data_dict[b'data'][3][:1024]
reshaped_image = np.reshape(img, (32, 32, 1), order='F')
plt.imshow(reshaped_image)
plt.show()

grayscale_images(merged_tests_dict)
merged_tests_dict = equalize_images(merged_tests_dict)
merged_tests_dict[b'data'] = [[float(num) for num in sublist] for sublist in merged_tests_dict[b'data']]
merged_tests_dict = normalise_image_pixel_values(merged_tests_dict)
print(len(merged_tests_dict[b'data'][3]))

# sample training image after normalisation and grayscaling
img = merged_data_dict[b'data'][3][:1024]
reshaped_image = np.reshape(img, (32, 32, 1), order='F')
plt.imshow(reshaped_image)
plt.show()

def zoom(mfernum1):
    zoom = iaa.Affine(scale=(1, 1.3)) #affine transformation preserves straight lines so zoom doesn't affect them
    mfernum1 = zoom.augment_image(mfernum1)
    return mfernum1

def pan(mfernum2):
    pan_func = iaa.Affine(translate_percent={"x":(-0.1, 0.1), "y":(-0.1, 0.1)})
    panned_image = pan_func.augment_image(mfernum2)
    return panned_image

def img_random_brightness(mfernum3):
    brightness = iaa.Multiply((0.2, 1.2)) #multiple image by 0.2 to make it darker, multiple by 1.2 to make it brighter
    mfernum3 = brightness.augment_image(mfernum3)
    return mfernum3

# augment training data
def random_augment(image):
    #image = mpimg.imread(image) 
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    return image

def batch_generator(image_paths, image_names, batch_size, is_training):
    while True:
        batch_img = []
        batch_names = []
        image_paths = np.asarray(image_paths)
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            img = np.reshape(image_paths[random_index][:1024], (32, 32, 1), order='F')
            names = image_names[random_index]
            if is_training: #can augment training data, but validation/test data we shouldn't augment
                img = random_augment(img)
            batch_img.append(img)
            batch_names.append(names)
        yield(np.asarray(batch_img), np.asarray(batch_names))


x_train_gen, y_train_gen = next(batch_generator(merged_data_dict[b'data'], merged_data_dict[b'labels'], 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(merged_tests_dict[b'data'], merged_tests_dict[b'labels'], 1, 0))

fig, axs = plt.subplots(1, 2, figsize=(15,10))
fig.tight_layout()
axs[0].imshow(x_train_gen[0])
axs[0].set_title("Training Image")
axs[1].imshow(x_valid_gen[0])
axs[1].set_title("Validation Image")
