import numpy as np
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical


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


def unite_the_trees(training_data):
    for i in range(len(training_data[b'labels'])):
        if training_data[b'labels'][i].find(b'_tree') != -1:
            training_data[b'labels'][i] = b'tree'


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

# unite tree labels

for value in combined_labels_dict[b'label_names']:
    print(value)

combined_labels_dict[b'label_names'].append(b'tree')
combined_labels_dict[b'label_names'].remove(b'maple_tree')
combined_labels_dict[b'label_names'].remove(b'oak_tree')
combined_labels_dict[b'label_names'].remove(b'palm_tree')
combined_labels_dict[b'label_names'].remove(b'pine_tree')
combined_labels_dict[b'label_names'].remove(b'willow_tree')

for value in combined_labels_dict[b'label_names']:
    print(value)


# Unite all tree classes under the label "tree"
unite_the_trees(merged_data_dict)
unite_the_trees(merged_tests_dict)

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


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess_images(img):
    img = np.reshape(img, (32, 32, 3), order='F')
    img = grayscale(img)
    img = equalize(img)
    img = img/255 # Normalization

    return img


# sample training image after normalisation and grayscaling
x_train, y_train = merged_data_dict[b'data'], merged_data_dict[b'labels']
x_test, y_test = merged_tests_dict[b'data'], merged_tests_dict[b'labels']

x_train = np.array(list(map(preprocess_images, x_train)))
random_image = x_train[3]
plt.imshow(random_image)
plt.show()

x_test = np.array(list(map(preprocess_images, x_test)))
random_image = x_test[3]
plt.imshow(random_image)
plt.show()


def zoom(mfernum1):
    zoom = iaa.Affine(scale=(1, 1.3))
    mfernum1 = zoom.augment_image(mfernum1)
    return mfernum1


def pan(mfernum2):
    pan_func = iaa.Affine(translate_percent={"x":(-0.1, 0.1), "y":(-0.1, 0.1)})
    panned_image = pan_func.augment_image(mfernum2)
    return panned_image


def img_random_brightness(mfernum3):
    brightness = iaa.Multiply((0.2, 1.2))
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


def create_augmented_images_for_cifar100_classes(image_paths, image_names):
    new_images = []
    new_names = []
    image_paths = np.asarray(image_paths, dtype=np.float32)
    for i in range(len(image_paths)):
        if image_names[i] in [b'baby', b'bicycle', b'boy', b'bus', b'cattle', b'fox', b'girl', b'lawn_mower', b'man', b'motorcycle', b'pickup_truck', b'rabbit', b'squirrel', b'tractor', b'train', b'woman']:
            img =image_paths[i]
            for j in range(9):
                img = random_augment(img)
                new_images.append(img)
                new_names.append(image_names[i])
        elif image_names[i] == b'tree':
            img = image_paths[i]
            img = random_augment(img)
            new_images.append(img)
            new_names.append(image_names[i])
    return new_images, new_names


new_images, new_names = create_augmented_images_for_cifar100_classes(x_train, y_train)

print(len(x_train))
print()
print(len(y_train))

x_train = np.concatenate((x_train, new_images))
y_train = np.concatenate((y_train, new_names))

print(len(x_train))
print()
print(len(y_train))

img = new_images[500]
plt.imshow(img)
plt.show()


def onehot_encode_labels(labels):
    encoder = OneHotEncoder(sparse_output=False)
    labels_reshaped = [[category] for category in labels]
    encoded_labels = encoder.fit_transform(labels_reshaped)

    return encoded_labels


print(y_train[0])
y_train = onehot_encode_labels(y_train)
print(y_train[0])
print(y_test[0])
y_test = onehot_encode_labels(y_test)
print(y_test[0])


# BUILDING OUR MODEL


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])


def plot_loss(model, X_train, y_train):
    print(X_train[0])
    print(len(X_train[0]))
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size = 1000, verbose=1, shuffle=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'validation_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()


def analyze_model(model, X_test, y_test, X_train, y_train):
    print(model.summary())
    plot_loss(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)


def lenet_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(24, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = lenet_model()
analyze_model(model, x_test, y_test, x_train, y_train)

