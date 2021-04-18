from random import shuffle
from sklearn.model_selection import train_test_split
import json
import pathlib
import tensorflow as tf
import pathlib
import os


class DataSetCreator():
    def __init__(self, data_dir):
        self.root_dir = pathlib.Path(data_dir)
        self.create_label_json()
        self.images, self.labels = self.extract_data_and_labels()

    def create_label_json(self):
        labels = [str(dirs).split('/')[-1] for dirs in list(self.root_dir.glob('*'))]
        self.label_dict = {}
        i = 0
        for label in labels:
            self.label_dict[label] = i
            i += 1

        print("writing labels the following to label map \n", self.label_dict)
        with open('./label_map.json', 'w') as lf:
            json.dump(self.label_dict, lf)

    def extract_data_and_labels(self):
        images = [str(i) for i in list(self.root_dir.glob('*/*.png'))]
        labels = [self.label_dict[image.split('/')[-2]] for image in images]
        return images, labels

    def create_iterator_dataset(self, dataset):
        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        iterator_init_op = iterator.initializer

        inputs = {'images': images, 'labels': labels,
                'iterator_init_op': iterator_init_op}
        return inputs

    def get_train_data(self):
        ds = tf.data.Dataset.from_tensor_slices(self.images, self.labels)

    def get_train_val_data(self):
        pass

    def _parse(self, image, label, size=(224, 224)):
        image = tf.io.read(image)


def batch_and_optimize(dataset, buffer_size=1000, batch_size=8):
    ds = dataset.cache()
    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def load_data(file_path):
    label = tf.strings.split(file_path, os.sep, result_type='RaggedTensor')[-2]
    return tf.io.read_file(file_path), label


def create_dataset(root_path, image_format=None, training=True, batch_size=8):
    path = pathlib.Path(root_path)
    num_samples = len(list(path.glob("*/*.{}".format(image_format))))
    labels = [os.path.basename(str(i)) for i in list(path.glob("*"))]

    if training:
        dataset = tf.data.Dataset.list_files(str(
            path/"*/*.{}".format(image_format))).map(load_data).shuffle(num_samples).batch(batch_size).prefetch(1)

    else:
        dataset = tf.data.Dataset.list_files(str(
            path/"*/*.{}".format(image_format))).shuffle(num_samples).map(load_data).batch(batch_size).prefetch(1)

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels,
              'iterator_init_op': iterator_init_op}
    return inputs


if __name__ == "__main__":
    dataset_path = "/media/maheshwaran.umapathy/thunder/datasets/opensource/cifar/cifar_10/train"
    # dataset = create_dataset(dataset_path, 'png')
    # print(dataset)
    ds = DataSetCreator(dataset_path)
