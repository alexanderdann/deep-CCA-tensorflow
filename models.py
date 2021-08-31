import tensorflow as tf

class MultilayerPerceptron(tf.keras.Model):
    def __init__(self, network_data, l_rate=.01, m_rate=.0, views=2):
        super(MultilayerPerceptron, self).__init__()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=l_rate,
                                                 momentum=m_rate)

        inputs, outputs = [], []
        for view_idx in range(views):
            tmp_input, tmp_output = self._connect(network_data, view_idx)
            inputs.append(tmp_input)
            outputs.append(tmp_output)

        self.MLP = tf.keras.Model(inputs=inputs, outputs=outputs, name='deepCCA')
        self.MLP.summary()
        self.MLP.compile()

    def _connect(self, network_data, view_idx):
        init_input = tf.keras.Input(shape=network_data[0][0],
                                     name=f'Input_Layer_{view_idx}')

        tmp_input = 0
        for counter, encoder_info in enumerate(network_data[1:], 1):
            name = f'View_{view_idx}_Layer_{counter}'
            if counter == 1:
                tmp_input = tf.keras.layers.Dense(encoder_info[0], activation=encoder_info[1], name=name)\
                    (init_input)
            else:
                tmp_input = tf.keras.layers.Dense(encoder_info[0], activation=encoder_info[1], name=name)\
                    (tmp_input)

        final_output = tmp_input

        return init_input, final_output

    def loss(self, view1, view2):
        V1 = tf.cast(view1, dtype=tf.float32)
        V2 = tf.cast(view2, dtype=tf.float32)

        r1 = tf.cast(1e-3, dtype=tf.float32)
        r2 = tf.cast(1e-3, dtype=tf.float32)
        eps = tf.cast(1e-4, dtype=tf.float32)

        assert V1.shape[0] == V2.shape[0]
        M = tf.constant(V1.shape[0], dtype=tf.float32)
        ddim = tf.constant(V1.shape[1], dtype=tf.int16)

        meanV1 = tf.reduce_mean(V1, axis=0, keepdims=True)
        meanV2 = tf.reduce_mean(V2, axis=0, keepdims=True)

        V1_bar = V1 - tf.tile(meanV1, [M, 1])
        V2_bar = V2 - tf.tile(meanV2, [M, 1])

        Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
        Sigma11 = tf.add(tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) , r1 * tf.eye(ddim))
        Sigma22 = tf.add(tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) , r2 * tf.eye(ddim))

        Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
        Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
        Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

        T = tf.matmul(tf.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv)
        TT = tf.matmul(tf.transpose(T), T)
        reg_TT = tf.add(TT, eps*tf.eye(ddim))
        corr = tf.linalg.trace(tf.linalg.sqrtm(reg_TT))
        return -corr


class CanonicalCorrelationAnalysis:
    def __init__(self, view1, view2):
        self.A, self.B, self.epsilon, self.omega, self.ccor = self._calculate(view1, view2)

    def getitems(self):
        return (self.A, self.B), (self.epsilon, self.omega, self.ccor)

    def _calculate(self, view1, view2):
        print('\n\n\n--------- CCA Start ---------\n\n')
        V1 = tf.cast(view1, dtype=tf.float32)
        V2 = tf.cast(view2, dtype=tf.float32)

        r1 = 1e-2
        r2 = 1e-2

        assert V1.shape[0] == V2.shape[0]
        M = tf.constant(V1.shape[0], dtype=tf.float32)
        ddim = tf.constant(V1.shape[1], dtype=tf.int16)

        V1_bar = V1 - tf.matmul(V1, tf.ones(shape=[ddim, ddim]))  # tf.tile(meanV1, [M, 1])
        V2_bar = V2 - tf.matmul(V2, tf.ones(shape=[ddim, ddim]))  # tf.tile(meanV2, [M, 1])

        Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
        Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + r1 * tf.eye(ddim)
        Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + r2 * tf.eye(ddim)

        Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
        Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
        Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

        C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)

        D, U, V = tf.linalg.svd(C, full_matrices=True)

        A = tf.matmul(tf.transpose(U), Sigma11_root_inv)
        B = tf.matmul(tf.transpose(V), Sigma22_root_inv)

        epsilon = tf.matmul(A, tf.transpose(V1_bar))
        omega = tf.matmul(B, tf.transpose(V2_bar))

        print("Canonical Correlations: " + str(D))
        print('\n\n--------- CCA End ---------')
        return A, B, epsilon, omega, D


