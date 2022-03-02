import tensorflow as tf
from tqdm import tqdm
import shutil
from models import MultilayerPerceptron
import os
from TwoChannelModel import TwoChannelModel
from correlation_analysis import CCA
import tf_summary as tfs

LOGPATH = f'{os.getcwd()}/LOG/TEST'
shutil.rmtree(LOGPATH)
os.makedirs(LOGPATH)

rhos = [0.9, 0.75, 0.0, 0.0, 0.0]
batch_size = 1024
samples = 1024
z_dim = 2
c_dim = 3
num_views = 2
epochs = 1000

assert z_dim == 2

# Generate data
data_model = TwoChannelModel(num_samples=1000)
# Possible modes for data_model: 'Gaussian' or 'Parabola'
y_1, y_2, Az_1, Az_2, z_1, z_2 = data_model('Gaussian', rhos)

# list containing all information about the each layer
MLP_layers = [(5, 'sigmoid'), (64, 'sigmoid'), (5, None)]

def train(MLP_layers, view1, view2, epochs):
    deepCCA_Class = MultilayerPerceptron(MLP_layers, 0.9, 0.0)
    deepCCA_Model = deepCCA_Class.MLP
    tf_view1 = tf.convert_to_tensor(view1, dtype=tf.float32)
    tf_view2 = tf.convert_to_tensor(view2, dtype=tf.float32)

    observations = view1.shape[0]
    num_channels = view1.shape[1]

    writer = tfs.create_grid_writer(LOGPATH, params=['deepCCA', rhos])
    for epoch in tqdm(range(epochs), desc='Epochs'):
        with tf.GradientTape() as tape:
            tape.watch([tf_view1, tf_view2])

            fy_1, fy_2 = deepCCA_Model([tf.transpose(tf_view1), tf.transpose(tf_view2)])
            c_loss = deepCCA_Class.loss(fy_1, fy_2)

        gradients = tape.gradient(c_loss, deepCCA_Model.trainable_variables)
        deepCCA_Class.optimizer.apply_gradients(zip(gradients, deepCCA_Model.trainable_variables))

        if epoch % 10 == 0:
            B1, B2, epsilon, omega, ccor = CCA(fy_1, fy_2, 5)
            if rhos[0] == 1:
                sim_v1 = tfs.compute_similarity_metric_v1(S=z_1[:2], U=0.5*(omega+epsilon))
            else:
                sim_v1 = 0
            sim_v2 = tfs.compute_similarity_metric_v1(S=z_1[:2], U=epsilon)
            sim_v3 = tfs.compute_similarity_metric_v1(S=z_2[:2], U=omega)
            dist = tfs.compute_distance_metric(S=z_1[:2], U=0.5 * (omega + epsilon)[:2])

            tfs.write_scalar_summary(
                writer=writer,
                epoch=epoch,
                list_of_tuples=[
                    (ccor[0], 'Canonical correlation/1'),
                    (ccor[1], 'Canonical correlation/2'),
                    (ccor[2], 'Canonical correlation/3'),
                    (ccor[3], 'Canonical correlation/4'),
                    (ccor[4], 'Canonical correlation/5'),
                    (c_loss, 'Loss/CCA'),
                    (dist, 'Performance Measures/Distance measure'),
                    (sim_v1, 'Performance Measures/Similarity measure'),
                    (sim_v2, 'Performance Measures/Similarity measure 1st view'),
                    (sim_v3, 'Performance Measures/Similarity measure 2nd view'),
                ]
            )

        if epoch % 2500 == 0 or epoch == 99999:
            tfs.write_PCC_summary(writer, epoch, z_1, z_2, epsilon, omega, 1000)

train(MLP_layers, y_1, y_2, epochs=epochs)

if __name__ == '__main__':
    print(tf.version)