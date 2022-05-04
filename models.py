import tensorflow as tf
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp
from sklearn.svm import LinearSVC as SVM
from sklearn.metrics import accuracy_score

        
def _create_mlp(input_dim, hidden_layers, shared_dim, view_idx, activation='sigmoid'):
    input_layer = tf.keras.Input(shape=(input_dim, ), name=f'view{view_idx}_input_layer')
    
    for layer_idx, layer_dim in enumerate(hidden_layers, start=1):
        if layer_idx == 1:
            tmp_hidden_layer = tf.keras.layers.Dense(layer_dim, activation, name=f'view_{view_idx}_hidden_layer_{layer_idx}')\
            (input_layer)
            
        else:
            tmp_hidden_layer = tf.keras.layers.Dense(layer_dim, activation, name=f'view_{view_idx}_hidden_layer_{layer_idx}')\
            (tmp_hidden_layer)
            
            
    output_layer = tf.keras.layers.Dense(shared_dim, activation='linear', name=f'view_{view_idx}_output_layer')\
                    (tmp_hidden_layer)
    
    return input_layer, output_layer
            
    
    
def build_deepCCA_model(input_dims, hidden_layers, shared_dim, activation, views=2, learning_rate=0.001, momentum_rate=0.0, display_model=False):
    '''
        Network Dimensions are set using input_dim : integer
    '''
    input_layers, output_layers = list(), list()
    
    for view_idx in range(views):
        input_layer, output_layer = _create_mlp(input_dims[view_idx], hidden_layers, shared_dim, view_idx, activation)
        input_layers.append(input_layer)
        output_layers.append(output_layer)

    model = tf.keras.Model(inputs=input_layers, outputs=output_layers, name='deepCCA')
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum_rate))
    model.summary()
    
    if display_model:
        plot_model(model, to_file='deepCCA model.png', show_shapes=True, show_layer_activations=True)

    return model


def compute_loss(view1, view2):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    r1 = tf.cast(1e-4, dtype=tf.float32)
    r2 = tf.cast(1e-2, dtype=tf.float32)
    eps = tf.cast(1e-5, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]

    M = tf.constant(V1.shape[0], dtype=tf.float32)
    ddim = tf.constant(V1.shape[1], dtype=tf.int16)

    meanV1 = tf.reduce_mean(V1, axis=0, keepdims=True)
    meanV2 = tf.reduce_mean(V2, axis=0, keepdims=True)

    V1_bar = V1 - tf.tile(meanV1, [M, 1])
    V2_bar = V2 - tf.tile(meanV2, [M, 1])

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.add(tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1), r1 * tf.eye(ddim))
    Sigma22 = tf.add(tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1), r2 * tf.eye(ddim))

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)
    T = tf.matmul(tf.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    TT = tf.matmul(tf.transpose(T), T)
    reg_TT = tf.add(TT, eps*tf.eye(ddim))
    corr = tf.linalg.trace(tf.linalg.sqrtm(reg_TT))
    return -corr


def compute_regularization(model, lambda_reg=1e-4):
    reg_term = 0
    for idx, trainable_var in enumerate(model.trainable_variables):
        reg_term += tf.norm(trainable_var, ord=2)

    return lambda_reg * reg_term


def compute_termination_score(train_data, train_labels, test_data, test_labels):
    assert train_data.shape[0] == train_labels.shape[0]
    assert test_data.shape[0] == test_labels.shape[0]
    svm_model = SVM(random_state=333)
    svm_model.fit(train_data.numpy(), train_labels)
    predictions = svm_model.predict(test_data.numpy())
    return accuracy_score(test_labels, predictions)


def CCA(view1, view2, pca_dim=None):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    r1 = tf.cast(1e-4, dtype=tf.float32)
    r2 = tf.cast(1e-4, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]
    M = tf.constant(V1.shape[0], dtype=tf.float32)
    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16)
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16)

    mean_V1 = tf.reduce_mean(V1, 0)
    mean_V2 = tf.reduce_mean(V2, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + r1 * tf.eye(ddim_1)
    Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + r2 * tf.eye(ddim_2)

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

    C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    D, U, V = tf.linalg.svd(C, full_matrices=True)

    A = tf.matmul(tf.transpose(U), Sigma11_root_inv)
    B = tf.matmul(tf.transpose(V), Sigma22_root_inv)

    epsilon = tf.matmul(A, tf.transpose(V1_bar))
    omega = tf.matmul(B, tf.transpose(V2_bar))

    return A, B, epsilon, omega, D


def PCC_Matrix(view1, view2, observations):
    assert tf.shape(view1)[1] == observations
    assert tf.shape(view1)[1] == tf.shape(view2)[1]
    calc_cov = tfp.stats.correlation(view1, view2, sample_axis=1, event_axis=0)

    return tf.math.abs(calc_cov), tf.shape(view1)[0], tf.shape(view2)[0]
