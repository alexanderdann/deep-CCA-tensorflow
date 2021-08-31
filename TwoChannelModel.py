import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import default_rng
np.random.seed(3333)
sns.set()
sns.set_style('white')
sns.set_context('paper')

class TwoChannelModel():
    def __init__(self, path, observations, mixing_dim, shared_dim, private_dim, mode, transformation=False, rhos=np.array([-1])):
        print("----- TwoChannelModel -----\n")

        self.path = path
        np.random.seed(3333)

        self._transform(observations, mixing_dim, shared_dim, private_dim, mode, transformation, rhos)

        print("Rows of Spatial Maps: " + str(len(self.A_x)))
        print("Columns of Spatial Maps: " + str(len(self.A_x[0])) + "\n")

        print("Number of Analyzed Brain Regions: " + str(len(self.S_x[0])))
        print("Number of Observations: " + str(len(self.S_x)) + "\n")

    def eval(self, batch_size, num_channels, t_min=-10, t_max=10):
        self.eval_data, self.test_sample = self._create_evaluation_data(batch_size, num_channels, t_min=-10, t_max=10)
        return self.eval_data, self.test_sample

    def getitems(self):
        return self.X, self.Y, self.S_x, self.S_y, self.created_rhos

    def _transform(self, observations, mixing_dim, shared_dim, private_dim, mode, nonlinearTr, rhos):
        num_comp = shared_dim + private_dim

        self.A_x = np.random.randn(num_comp, mixing_dim)
        self.A_y = np \
            .random.randn(num_comp, mixing_dim)

        self.mv_samples = []
        for num in range(num_comp):
            mean = np.array([0, 0])

            cov = np.array([[1, 0],
                            [0, 1]])

            multivar_sample = np.random.multivariate_normal(mean, cov, size=mixing_dim, check_valid='warn', tol=1e-10)
            self.mv_samples.append(multivar_sample.T)

        for i in range(num_comp):
            self.A_x[i] = self.mv_samples[i][0]
            self.A_y[i] = self.mv_samples[i][1]

        if mode == 'Gaussian':
            self.mix = 'Gaussian'


            assert len(rhos) == shared_dim + private_dim

            self.S_x = np.random.randn(observations, num_comp)
            self.S_y = np.random.randn(observations, num_comp)

            self.mv_samples = []
            for num in range(len(rhos)):
                mean = np.array([0, 0])

                cov = np.array([[1, rhos[num]],
                                [rhos[num], 1]])

                multivar_sample = np.random.multivariate_normal(mean, cov, size=observations, check_valid='warn',
                                                                tol=1e-10)
                self.mv_samples.append(multivar_sample.T)

            for i in range(len(rhos)):
                self.S_x.T[i] = self.mv_samples[i][0]
                self.S_y.T[i] = self.mv_samples[i][1]

        elif mode == 'Parabola':
            self.mix = 'Parabola'

            repr1, repr2 = self._gen_parabola(observations)

            shared_x = np.array([repr1, repr2])
            shared_y = np.copy(shared_x)
            #shared_x = np.tile(repr1, (shared_dim, 1))
            #shared_y = np.tile(repr2, (shared_dim, 1))

            plt.scatter(shared_x[0], shared_x[1])
            plt.xlabel('$\mathrm{dimension1}$', fontsize='18')
            plt.ylabel('$\mathrm{dimension2}$', fontsize='18')
            plt.show()

            private_x = np.random.random((private_dim, repr1.shape[0]))
            private_y = np.random.random((private_dim, repr2.shape[0]))

            self.S_x = np.concatenate([shared_x, private_x], axis=0).T
            self.S_y = np.concatenate([shared_y, private_y], axis=0).T

        else:
            print('ERROR\n')
            raise ValueError

        X = np.dot(self.S_x, self.A_x).T
        Y = np.dot(self.S_y, self.A_y).T

        self.created_rhos = self._PCC(self.S_x, self.S_y)

        if nonlinearTr == True:
            _sigmoid = lambda x: np.array([1/(1 + np.exp(-x_i)) for x_i in x])

            for i in range(num_comp):
                if i == 0:
                    X[i] = 3 * _sigmoid(X[i]) + 0.1 * X[i]
                    Y[i] = 5 * np.tanh(Y[i]) + 0.2 * Y[i]

                elif i == 1:
                    X[i] = 5 * _sigmoid(X[i]) + 0.2 * X[i]
                    Y[i] = 2 * np.tanh(Y[i]) + 0.1 * Y[i]

                elif i == 2:
                    X[i] = 0.2 * np.exp(X[i])
                    Y[i] = 0.1 * Y[i]**3 + Y[i]

                elif i == 3:
                    X[i] = -4 * _sigmoid(X[i]) - 0.3 * X[i]
                    Y[i] = -5 * np.tanh(Y[i]) - 0.4 * Y[i]

                elif i == 4:
                    X[i] = -3 * _sigmoid(X[i]) + 0.2 * X[i]
                    Y[i] = -6 * np.tanh(Y[i]) - 0.3 * Y[i]

                else:
                    break

        mix = self.mix

        self.X = X.T
        self.Y = Y.T

        plt.rcParams.update({'figure.figsize': (5, 4)})
        # plt.suptitle('Relationship between True Sources $\mathbf{S}_{\mathrm{X}}$ and $\mathbf{S}_{\mathrm{Y}}$', fontweight='bold', fontsize=19)
        title = 'Transformation: ' + mix
        plt.title(title, fontsize='14')
        plt.ylabel('$\mathbf{X}$', fontweight='bold', fontsize='18')
        plt.xlabel('$\mathbf{Y}$', fontweight='bold', fontsize='18')
        legend = plt.scatter(self.X.T[0], self.Y.T[0], c='black', marker='.')
        # legend.set_label('rhos=[1, 1, 1]')
        # legend = plt.scatter(self.TC_x, self.test, c='black', marker='.')
        # legend.set_label('rhos=[0.95, 0.95, 0.95]')
        # plt.legend()
        plt.xlim(-5, 5)
        plt.tight_layout()
        full_path = self.path + '/' + 'GENSIG.png'
        plt.savefig(full_path)
        plt.show(block=False)
        plt.close('all')

        print("Generated Signal of Dimensions {0} X {1} \n".format(len(self.X), len(self.X[0])))

    def _PCC(self, TC_x, TC_y):
        calc_cov = []

        print(f'self.TC_y {TC_y.shape}')

        for i in range(len(TC_y.T)):
            sigma_y = np.sqrt(np.array(sum([y ** 2 for y in TC_y.T[i]])) / len(TC_y))
            sigma_x = np.sqrt(np.array(sum([x ** 2 for x in TC_x.T[i]])) / len(TC_x))
            calc_cov.append(np.dot(TC_x.T[i], TC_y.T[i]) / (len(TC_x) * sigma_y * sigma_x))

        calc_cov = np.sort(calc_cov)
        calc_cov = calc_cov[::-1].copy()

        for cor in range(len(calc_cov)):
            if calc_cov[cor] > 1:
                calc_cov[cor] = 1
            elif calc_cov[cor] < 0:
                calc_cov[cor] = 0

        print(f'That are the computed correlations: {calc_cov}')

        return calc_cov


    def _gen_parabola(self, observations):
        x = np.linspace(-1, 1, observations)
        f_x = x**2

        return x, f_x

    def _create_evaluation_data(self, batch_size, data_dim, t_min=-10, t_max=10):
        test_sample = np.linspace(t_min, t_max, batch_size)
        tile_dim = [data_dim, 1]
        view1 = np.tile(np.copy(test_sample[None]), tile_dim)
        view2 = np.copy(view1)
        _sigmoid = lambda x: np.array([1 / (1 + np.exp(-x_i)) for x_i in x])

        for i in range(data_dim):
            if i == 0:
                view1[i] = 3 * _sigmoid(view1[i]) + 0.1 * view1[i]
                view2[i] = 5 * np.tanh(view2[i]) + 0.2 * view2[i]

            elif i == 1:
                view1[i] = 5 * _sigmoid(view1[i]) + 0.2 * view1[i]
                view2[i] = 2 * np.tanh(view2[i]) + 0.1 * view2[i]

            elif i == 2:
                view1[i] = 0.2 * np.exp(view1[i])
                view2[i] = 0.1 * view2[i] ** 3 + view2[i]

            elif i == 3:
                view1[i] = -4 * _sigmoid(view1[i]) - 0.3 * view1[i]
                view2[i] = -5 * np.tanh(view2[i]) - 0.4 * view2[i]

            elif i == 4:
                view1[i] = -3 * _sigmoid(view1[i]) - 0.2 * view1[i]
                view2[i] = -6 * np.tanh(view2[i]) - 0.3 * view2[i]

            else:
                break

        #views_concat = np.concatenate([view1, view2], axis=0)
        #final_data = np.array([views_concat[i, :][None] for i in range(2 * data_dim)])

        return [view1.T, view2.T], test_sample