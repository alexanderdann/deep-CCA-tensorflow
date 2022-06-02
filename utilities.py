import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.utils import gen_batches
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC as SVM
from sklearn.metrics import accuracy_score

"""
    Used to suppress the warning related to very big dataset in the SVM and track_test_progress
"""
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def batch_data(data, batch_size):
    channels, samples = data.shape
    num_batches = samples // batch_size

    tmp = np.zeros(shape=(num_batches, batch_size, channels), dtype=np.float32)
    for batch_idx, indeces in enumerate(gen_batches(samples, batch_size)):
        tmp[batch_idx] = data[:, indeces].T
        
    assert num_batches_1 == num_batches_2
    return tf.convert_to_tensor(tmp, dtype=tf.float32), num_batches


def prepare_data(data, batch_size):
    '''
        Used to make batches of data. Either batching data into multiple batches which makes
        use of 'batch_data()' (if-case) or just converts to tensor and adds one dimension (else-case)
    '''
    proc_data, batches = list(), dict()
    
    for view_idx, view_data in enumerate(data):
        tmp_data = dict()
        
        for key in ['train', 'validation', 'test']:
            if batch_size is not None:
                batch, num_batches = batch_data(view_data[key]['data'], batch_size)

            else:
                batch, num_batches = tf.convert_to_tensor(view_data[key]['data'].T, dtype=tf.float32)[None], 1

            tmp_data[key] = batch
            batches[key] = num_batches
            
        proc_data.append(tmp_data)
                
    return proc_data, batches


def load_mlsp_data():
    labels = pd.read_csv('mlsp-2014-mri/Train/train_labels.csv').to_numpy()[:, 1]
    labeled_FNC = pd.read_csv('mlsp-2014-mri/Train/train_FNC.csv').to_numpy()[:, 1:]
    labeled_SBM = pd.read_csv('mlsp-2014-mri/Train/train_SBM.csv').to_numpy()[:, 1:]
    
    unlabeled_FNC = pd.read_csv('mlsp-2014-mri/Test/test_FNC.csv').to_numpy()[:, 1:]
    unlabeled_SBM = pd.read_csv('mlsp-2014-mri/Test/test_SBM.csv').to_numpy()[:, 1:]
    
    
    data_FNC = list()
    data_SBM = list()
    
    for train_idx, tmp_idx in StratifiedKFold(n_splits=4).split(np.zeros(len(labels)), labels):
        for val_idx, test_idx in StratifiedKFold(n_splits=2).split(np.zeros(len(labels[tmp_idx])), labels[tmp_idx]):
            for modality, data in [('FNC', labeled_FNC), ('SBM', labeled_SBM)]:
                
                X_train, y_train = data[train_idx], labels[train_idx]
                X_test, y_test = data[tmp_idx][test_idx], labels[tmp_idx][test_idx]
                X_val, y_val = data[tmp_idx][val_idx], labels[tmp_idx][val_idx]

                if modality == 'FNC':
                    data_dict = {'unlabeled': {'data': unlabeled_FNC.T, 'labels': None},
                                 'train': {'data': X_train.T, 'labels': y_train},
                                 'validation': {'data': X_val.T, 'labels': y_val},
                                 'test': {'data': X_test.T, 'labels': y_test}}
                    data_FNC.append(data_dict)

                elif modality == 'SBM':
                    data_dict = {'unlabeled': {'data': unlabeled_SBM.T, 'labels': None},
                                 'train': {'data': X_train.T, 'labels': y_train},
                                 'validation': {'data': X_val.T, 'labels': y_val},
                                 'test': {'data': X_test.T, 'labels': y_test}}
                    data_SBM.append(data_dict)
        
        
    return data_FNC, data_SBM, labels


def compute_termination_score(train_data, train_labels, test_data, test_labels):
    assert train_data.shape[0] == train_labels.shape[0]
    assert test_data.shape[0] == test_labels.shape[0]
    svm_model = SVM(random_state=333)
    svm_model.fit(train_data.numpy(), train_labels)
    predictions = svm_model.predict(test_data.numpy())
    return accuracy_score(test_labels, predictions)


@ignore_warnings(category=ConvergenceWarning)
def track_test_progress_v2(model, data, test_labels, validation_labels):
    test_fy_1, test_fy_2 = model([data[0]['test'][0], data[1]['test'][0]])
    test_fy = tf.concat([test_fy_1, test_fy_2], axis=1)
    
    val_fy_1, val_fy_2 = model([data[0]['validation'][0], data[1]['validation'][0]])
    val_fy = tf.concat([val_fy_1, val_fy_2], axis=1)
    
    return compute_termination_score(val_fy, validation_labels, test_fy, test_labels)


@ignore_warnings(category=ConvergenceWarning)
def track_validation_progress(model, data, train_labels, validation_labels):
    train_fy_1, train_fy_2 = model([data[0]['train'][0], data[1]['train'][0]])
    train_fy = tf.concat([train_fy_1, train_fy_2], axis=1)
    
    val_fy_1, val_fy_2 = model([data[0]['validation'][0], data[1]['validation'][0]])
    val_fy = tf.concat([val_fy_1, val_fy_2], axis=1)
    
    return compute_termination_score(train_fy, train_labels, val_fy, validation_labels)


@ignore_warnings(category=ConvergenceWarning)
def track_test_progress(model, data, train_labels, validation_labels, test_labels):
    train_fy_1, train_fy_2 = model([data[0]['train'][0], data[1]['train'][0]])
    val_fy_1, val_fy_2 = model([data[0]['validation'][0], data[1]['validation'][0]])
    
    train_val_1 = tf.concat([train_fy_1, val_fy_1], axis=0)
    train_val_2 = tf.concat([train_fy_2, val_fy_2], axis=0)
    train_val_concat = tf.concat([train_val_1, train_val_2], axis=1)
    
    test_fy_1, test_fy_2 = model([data[0]['test'][0], data[1]['test'][0]])
    test_concat = tf.concat([test_fy_1, test_fy_2], axis=1)
    
    train_val_labels = tf.concat([train_labels, validation_labels], axis=0)
    
    return compute_termination_score(train_val_concat, train_val_labels, test_concat, test_labels)


def test_raw_single(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    view_1 = tf.concat([train_data[0].T, validation_data[0].T], axis=0)
    view_2 = tf.concat([train_data[1].T, validation_data[1].T], axis=0)
    
    labels_1 = tf.concat([train_labels[0], validation_labels[0]], axis=0)
    labels_2 = tf.concat([train_labels[1], validation_labels[1]], axis=0)
    
    score_view1 = compute_termination_score(view_1, labels_1, tf.convert_to_tensor(test_data[0].T), test_labels[0])
    score_view2 = compute_termination_score(view_2, labels_2, tf.convert_to_tensor(test_data[1].T), test_labels[1])
    
    return [score_view1, score_view2]


@ignore_warnings(category=ConvergenceWarning)
def test_raw(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    
    print(train_data[0].shape, validation_data[0].shape, test_data[0].shape)
    
    view_1 = tf.concat([train_data[0], validation_data[0]], axis=1)
    view_2 = tf.concat([train_data[1], validation_data[1]], axis=1)
    print('###')
    print(tf.shape(view_1), tf.shape(view_2))
    print('###')
    view_concat = tf.concat([view_1, view_2], axis=0)
    
    train_labels = tf.concat([train_labels, validation_labels], axis=0)

    test_concat = tf.concat([test_data[0], test_data[1]], axis=0)
    
    print(tf.shape(test_concat), tf.shape(view_concat), tf.shape(train_labels), tf.shape(test_labels))
    
    return compute_termination_score(tf.transpose(view_concat), train_labels, tf.transpose(test_concat), test_labels)
    
