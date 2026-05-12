# Import library
import numpy as np
import matplotlib.pyplot as plt

# Buat kelas Backpropagation
class Backpropagation:

    # Konstruktor
    def __init__(self, alpha, epoch, target_error):

        self.alpha = alpha
        self.epoch = epoch
        self.target_error = target_error

        self.n_input = 2
        self.n_hidden = 2
        self.n_output = 1

        self.w_hidden = np.random.rand(self.n_input, self.n_hidden)
        self.b_hidden = np.random.rand(1, self.n_hidden)

        self.w_output = np.random.rand(self.n_hidden, self.n_output)
        self.b_output = np.random.rand(1, self.n_output)

        # Fungsi sigmoid bipolar / tanh
    def bi_sigmoid(self, x):
        return np.tanh(x)

    # Turunan sigmoid bipolar
    def deriv_bi_sigmoid(self, x):
        return 1 - x**2
    
        # Fungsi membuat simulasi perbaikan bobot dan bias
    def plot_error(self, x, epoch):

        plt.plot(
            range(1, epoch + 1),
            x,
            linestyle='-',
            color='b',
            label='Error'
        )

        final_error = x[-1]

        plt.annotate(
            f'Epoch {epoch}, Error: {final_error:.4f}',
            xy=(epoch, final_error),
            xytext=(epoch - len(x) * 0.2, final_error + 0.05),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=10,
            color='red'
        )

        plt.title('Perbaikan Error Setiap Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Sum Square Error(SSE)')

        plt.grid(True)
        plt.legend()
        plt.show()

        # Fungsi utama Backpropagation
    def fit(self, X, t):

        errors_per_epoch = []

        with open("hasilBackpropagation.txt", "w") as f:

            for epoch in range(self.epoch):

                total_error = 0
                output = np.array([])

                for xi, target in zip(X, t):

                    # FORWARD PROPAGATION

                    h_in = np.dot(xi, self.w_hidden) + self.b_hidden

                    h = self.bi_sigmoid(h_in)

                    y_in = np.dot(h, self.w_output) + self.b_output

                    y = self.bi_sigmoid(y_in)

                    output = np.append(output, y)

                    # BACKWARD PROPAGATION

                    error = target - y

                    total_error += np.sum(error**2)

                    d_y = error * self.deriv_bi_sigmoid(y)

                    error_h = np.dot(d_y, self.w_output.T)

                    d_h = error_h * self.deriv_bi_sigmoid(h)

                    # Update output layer
                    self.w_output += np.dot(h.T, d_y) * self.alpha

                    self.b_output += (
                        np.sum(d_y, axis=0, keepdims=True)
                        * self.alpha
                    )

                    # Update hidden layer
                    self.w_hidden += (
                        np.dot(xi.reshape(2,1), d_h)
                        * self.alpha
                    )

                    self.b_hidden += (
                        np.sum(d_h, axis=0, keepdims=True)
                        * self.alpha
                    )

                average_error = total_error / len(X)

                errors_per_epoch.append(average_error)

                if (
                    average_error < self.target_error
                    or epoch + 1 == self.epoch
                ):

                    self.plot_error(errors_per_epoch, epoch + 1)

                    break