import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import cv2
from sklearn.utils import gen_batches
import shutil
from models import MultilayerPerceptron
from correlation_analysis import CCA

from tensorboard_utillities import write_scalar_summary, write_image_summary, write_PCC_summary, write_gradients_summary_mean, write_poly
from tensorboard_utillities import create_grid_writer

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#default_mnist = tfds.load('mnist', as_supervised=True, data_dir='/Users/alexander/Documents/Uni/Work/Repositories/NMCA/tensorflow_datasets/')
#rotated_mnist = tfds.load('mnist_corrupted/rotate', as_supervised=True, data_dir='/Users/alexander/Documents/Uni/Work/Repositories/NMCA/tensorflow_datasets/')
#noisy_mnist = tfds.load('mnist_corrupted/impulse_noise', as_supervised=True, data_dir='/Users/alexander/Documents/Uni/Work/Repositories/NMCA/tensorflow_datasets/')

default_mnist = tfds.load('mnist', as_supervised=True, data_dir=f'{os.getcwd()}/tensorflow_datasets/')
rotated_mnist = tfds.load('mnist_corrupted/rotate', as_supervised=True, data_dir=f'{os.getcwd()}/tensorflow_datasets/')
noisy_mnist = tfds.load('mnist_corrupted/impulse_noise', as_supervised=True, data_dir=f'{os.getcwd()}/tensorflow_datasets/')

def prepare_dataset(data):
    tmp_test, tmp_train = list(), list()
    for image, label in tqdm(tfds.as_numpy(data['test']), desc='Loading Part I of Data'):
        fimg = cv2.normalize(image[0:28:2, 0:28:2], None, 0, 1, cv2.NORM_MINMAX)
        tmp_test.append(fimg.flatten())

    for image, label in tqdm(tfds.as_numpy(data['train']), desc='Loading Part II of Data'):
        fimg = cv2.normalize(image[0:28:2, 0:28:2], None, 0, 1, cv2.NORM_MINMAX)
        tmp_train.append(fimg.flatten())

    return np.array(tmp_test).T, np.array(tmp_train).T[:, :10000], np.array(tmp_train).T[:, 10000:]


#d_mnist_test, d_mnist_val, d_mnist_train = prepare_dataset(default_mnist)
r_mnist_test, r_mnist_val, r_mnist_train = prepare_dataset(rotated_mnist)
n_mnist_test, n_mnist_val, n_mnist_train = prepare_dataset(noisy_mnist)
print(r_mnist_test.shape, r_mnist_train.shape)

view1 = r_mnist_train
view2 = n_mnist_train
desc = 'ROTATED & NOISY'

LOGPATH = f'{os.getcwd()}/LOGY/{desc}'
MODELSPATH = f'{os.getcwd()}/MODELSY/{desc}'

MLP_layers = [(196, 'sigmoid'), (1024, 'sigmoid'), (1024, 'sigmoid'), (1024, 'sigmoid'), (196, None)]


def train(MLP_layers, view1, view2, epochs, batch_size, log_path, model_path, shared_dim=25):
    deepCCA_Class = MultilayerPerceptron(MLP_layers, 0.9, 0.0)
    model = deepCCA_Class.MLP

    channels, samples = view1.shape
    num_batches = samples // batch_size

    tmp_1, tmp_2 = list(), list()
    for batch_idx in gen_batches(samples, batch_size):
        tmp_1.append(view1[:, batch_idx].T)
        tmp_2.append(view2[:, batch_idx].T)
    y_1 = tf.convert_to_tensor(tmp_1, dtype=tf.float32)
    y_2 = tf.convert_to_tensor(tmp_2, dtype=tf.float32)

    observations = view1.shape[0]
    num_channels = view1.shape[1]

    writer = create_grid_writer(root_dir=log_path, params=['deepCCA', '1st run'])
    for epoch in tqdm(range(epochs), desc='Epochs'):
        losses, cca_losses, intermediate_outputs = list(), list(), list()

        for batch_idx in range(num_batches):
            batch_y1, batch_y2 = y_1[batch_idx], y_2[batch_idx]

            with tf.GradientTape() as tape:
                tape.watch([batch_y1, batch_y2])

                fy_1, fy_2 = model([batch_y1, batch_y2])
                c_loss = deepCCA_Class.loss(fy_1, fy_2)

            gradients = tape.gradient(c_loss, model.trainable_variables)
            deepCCA_Class.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch % 25 == 0:
                tmp = list()
                for batch_idx in range(num_batches):
                    batched_fy_1, batched_fy_2 = intermediate_outputs[batch_idx]
                    B1, B2, epsilon, omega, ccor = CCA(batched_fy_1, batched_fy_2, shared_dim)
                    tmp.append(ccor)

                avg_ccor = tf.math.reduce_mean(tmp, axis=0)
                static_part = [(tf.math.reduce_mean(losses), 'Loss/Total'),
                               (tf.math.reduce_mean(cca_losses), 'Loss/CCA'),]

                dynamic_part = [(cval, f'Canonical correlation/{idx})') for idx, cval in enumerate(avg_ccor)]
                write_scalar_summary(
                    writer=writer,
                    epoch=epoch,
                    list_of_tuples=static_part + dynamic_part
                )

            if epoch % 250 == 0:
                try:
                    os.makedirs(model_path)
                except FileExistsError:
                    print('MODELS PATH exists, saving data.')
                finally:
                    model.save(f'{model_path}/SharedDim-{shared_dim}-BatchSize-{batch_size}/model.tf', overwrite=True)
                    with open(f'{model_path}/SharedDim-{shared_dim}-BatchSize-{batch_size}/modellog.txt', 'a+') as f:
                        f.write(f'Saving model at epoch {epoch}\n')


train(MLP_layers, r_mnist_train, n_mnist_train, 2000, 5000, log_path=LOGPATH, model_path=MODELSPATH)


if __name__ == '__main__':
    print(tf.version)