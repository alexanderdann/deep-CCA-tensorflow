import tensorflow as tf
from models import MultilayerPerceptron, CanonicalCorrelationAnalysis
import time
import matplotlib.pyplot as plt
import os
from TwoChannelModel import *
import numpy as np

keys = time.asctime(time.localtime(time.time())).split()

# Please change your folder
path = '/Users/alexander/Documents/Uni/Work/NMCA/Simulation/' + str('-'.join(keys[0:3]))


try:
    os.makedirs(path)
    print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
except:
    pass

rhos = [0.9, 0.75, 0.0, 0.0, 0.0]
batch_size = 1024
samples = 1024
z_dim = 2
c_dim = 3
num_views = 2
epochs = 300

assert z_dim == 2

TCM = TwoChannelModel(
    path=path,
    observations=samples,
    mixing_dim=int(z_dim + c_dim),
    shared_dim=z_dim,
    private_dim=c_dim,
    mode='Parabola',
    transformation=True,
    rhos=rhos)

X, Y, S_x, S_y, created_rhos = TCM.getitems()

# list containing all information about the each layer
MLP_layers = [(5, 'sigmoid'), (128, 'sigmoid'), (128, 'sigmoid'), (5, None)]


def train(MLP_layers, data, epochs):
    deepCCA_Class = MultilayerPerceptron(MLP_layers, 0.9, 0.0)
    deepCCA_Model = deepCCA_Class.MLP
    tf_data = tf.convert_to_tensor(data, dtype=tf.float32)

    observations = data[0].shape[0]
    num_channels = data[0].shape[1]

    losses = []
    for epoch in range(epochs):
        print(f'\n######## Epoch {epoch + 1}/{epochs} ########')
        with tf.GradientTape() as tape:
            tape.watch(tf_data)

            output1, output2 = deepCCA_Model([tf_data[0], tf_data[1]])
            c_loss = deepCCA_Class.loss(output1, output2)
            losses.append(c_loss)
        print(f'Loss: {c_loss}')
        gradients = tape.gradient(c_loss, deepCCA_Model.trainable_variables)
        deepCCA_Class.optimizer.apply_gradients(zip(gradients, deepCCA_Model.trainable_variables))

    output1, output2 = deepCCA_Model([tf_data[0], tf_data[1]])
    _, cca_data = CanonicalCorrelationAnalysis(output1, output2).getitems()
    est1 = cca_data[0]
    est2 = cca_data[1]

    plt.scatter(est1[0], est1[1])
    plt.title(r'Estimated Sources 1')
    plt.show()

    plt.scatter(est2[0], est2[1])
    plt.title(r'Estimated Sources 2')
    plt.show()

    #eval_data_np, test_sample = TCM.eval(observations, num_channels)
    #eval_data_tf = tf.convert_to_tensor(eval_data_np, dtype=tf.float32)




train(MLP_layers, (X, Y), epochs=epochs)

if __name__ == '__main__':
    print(tf.version)