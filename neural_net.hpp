#ifndef NEURAL_NET_HPP_
#define NEURAL_NET_HPP_

#include <armadillo>
#include <config.h>
#include <string>
#include "util.hpp"
#include "mnist.hpp"
#include "gemm.hpp"

#include <algorithm>
#include <random>
#include <vector>
#include <numeric>

// Apply sigmoid to all elements
template<typename T>
void sigmoid(T& out) {
    out.transform([](typename T::elem_type val) {
        return (1.0 / (1.0 + exp (-1.0 * val)));
    });
}

/**
 * Calculate the weight gradient for a fully connected layer
 *
 * @param delta         Delta activations from the previous layer
 *                      (batch size x output features)
 * @param activations   Input activations to this layer
 *                      (batch size x input features)
 * @param bias          Bias matrix (batch_size x 1)
 * @param batch_size    Batch size
 * @param d_theta       Delta weights (output features x input features)
 * @param d_theta_bias  Delta bias weights (output features x 1)
 */
template<typename T>
void fc_gradient_weight(const T& delta, const T& activations, const T& bias,
    const typename T::elem_type batch_size, T& d_theta, T& d_theta_bias
) {
    using elem_t = typename T::elem_type;

    // d_theta = delta.t() * activations
    gemm<Trans, NoTrans>(
        delta.n_cols, activations.n_cols, delta.n_rows,
        elem_t(1) / batch_size,
        delta.memptr(), delta.n_rows,
        activations.memptr(), delta.n_rows,
        elem_t(0),
        d_theta.memptr(), delta.n_cols
    );

    // d_theta_bias = delta.t() * bias / batch_size;
    // Can also be implemented by summing down the columns
    // d_theta2_bias = sum(delta3, 0).t() / batch_size;
    gemm<Trans, NoTrans>(
        delta.n_cols, bias.n_cols, delta.n_rows,
        elem_t(1) / batch_size,
        delta.memptr(), delta.n_rows,
        bias.memptr(), delta.n_rows,
        elem_t(0),
        d_theta_bias.memptr(), delta.n_cols
    );
}

/**
 * Add regularization to fc_gradient_weight results
 */
template<typename T>
void fc_regularize(const T& theta, const T& theta_bias,
    const typename T::elem_type& lambda,
    const typename T::elem_type& batch_size, T& d_theta, T& d_theta_bias
) {
    using elem_t = typename T::elem_type;

    // d_theta += (lambda / batch_size) * theta;
    axpy(theta.n_rows * theta.n_cols, lambda / batch_size, theta.memptr(), 0,
        d_theta.memptr(), 0);
    // d_theta_bias += (lambda / batch_size) * theta_bias;
    axpy(d_theta_bias.n_rows * d_theta_bias.n_cols, lambda / batch_size,
        theta_bias.memptr(), 0, d_theta_bias.memptr(), 0);
}

/**
 * Calculate the gradient for the activations between two layers
 *
 * @param delta
 * @param theta
 * @param activation
 * @param result
 */
template<typename T>
void fc_gradient_activation(const T& delta, const T& theta,
    const T& activation, T& result
) {
    using elem_t = typename T::elem_type;
    using mat_t = T;

    // result = delta * theta % activation % (1 - activation);
    // % is an element-wise multiplication operator

    // result = delta * theta;
    gemm<NoTrans, NoTrans>(
        delta.n_rows, theta.n_cols, delta.n_cols,
        elem_t(1),
        delta.memptr(), delta.n_rows,
        theta.memptr(), delta.n_cols,
        elem_t(0),
        result.memptr(), delta.n_rows
    );

    // result = result % activation % (1 - activation);
    elem_t one = 1;
    for (size_t idx = 0; idx < result.n_rows * result.n_cols; ++idx) {
        elem_t act = activation[idx];
        result[idx] *= (act * (one - act));
    }
}

/**
 * Fully connected layer forward propogation
 *
 * @param activations  (batch size x input features) matrix
 * @param theta        (output features x input features) matrix
 * @param bias         (batch size x 1) matrix of 1s
 * @param theta_bias   (output features x 1) matrix of bias weights
 * @param result       (batch_size x output features) matrix
 */
template<typename T>
void fc_forward(const T& activations, const T& theta, const T& bias,
    const T& theta_bias, T& result
) {
    using elem_t = typename T::elem_type;

    // result = activations * theta.t();
    gemm<NoTrans, Trans>(
        activations.n_rows, theta.n_rows, activations.n_cols,
        elem_t(1),
        activations.memptr(), activations.n_rows,
        theta.memptr(), theta.n_rows,
        elem_t(0),
        result.memptr(), result.n_rows
    );

    // result += bias * theta_bias.t();
    gemm<NoTrans, Trans>(
        bias.n_rows, theta_bias.n_rows, bias.n_cols,
        elem_t(1),
        bias.memptr(), bias.n_rows,
        theta_bias.memptr(), theta_bias.n_rows,
        elem_t(1),
        result.memptr(), result.n_rows
    );
}

/**
 * Compute the delta between the labeled data and the predictions.
 * Dimensions of predictions, labels and delta must match
 *
 * @param predictions  The predictions
 * @param labels       The labels
 * @param delta        Delta between the predictions and labels
 */
template<typename T>
void delta_labels(const T& predictions, const T& labels, T& delta) {
    using elem_t = typename T::elem_type;
    // Activation delta from labeled data to current set of predictions
    // delta = predictions - labels
    // rewrite as:
    // delta = -labels + predictions
    const size_t sz = delta.n_rows * delta.n_cols;
    copy(sz, predictions.memptr(), 1, delta.memptr(), 1);
    axpy(sz, elem_t(-1), labels.memptr(), 1, delta.memptr(), 1);
}

/**
 * Simple two-layer neural network for the MNIST data set
 *
 * \code
 * auto train_net = neural_net<float>(train_data, 0.1);
 * \endcode
 */
template<typename T = float>
struct neural_net {
    /// The element type, either float or double
    using elem_t = T;
    /// Matrix type
    using mat_t = arma::Mat<elem_t>;
    /// Matrix subview type
    using subview_t = arma::subview<elem_t>;
    /// MNIST type
    using mnist = mnist<elem_t>;

    /// Input MNIST data
    const mnist input;
    /// Regularization parameter, set to 0 for no regularization
    const elem_t lambda;
    /// Weights for layer 1
    mat_t theta1;
    /// Weights for layer 2
    mat_t theta2;
    /// Bias weights for layer 1
    mat_t theta1_bias;
    /// Bias weights for layer 2
    mat_t theta2_bias;
    /// Input activations
    mat_t a1;
    /// Output activations from layer 1
    mat_t a2;
    /// Output activations from layer 2
    mat_t a3;
    /// One-hot vector for labels
    mat_t yy;
    /// Gradient for layer 1 weights
    mat_t d_theta1;
    /// Gradient for layer 2 weights
    mat_t d_theta2;
    /// Gradient for layer 1 bias weights
    mat_t d_theta1_bias;
    /// Gradient for layer 2 bias weights
    mat_t d_theta2_bias;
    /// Gradient from output to labels
    mat_t delta3;
    /// Gradient from layer 2 to layer 1 activations
    mat_t delta2;
    /// Colvec of 1s for the bias neurons
    mat_t bias;

    /**
     * Constructor
     *
     * @param input_   MNIST inut data
     * @param lambda_  lambda value for regularization. Set to 0 for no
     *                 regularization
     */
    neural_net(const mnist& input_, elem_t lambda_ = 1)
    : input(input_),
      lambda(lambda_) {
        using rowvec = arma::Row<elem_t>;
        using arma::zeros;
        using arma::ones;
        using arma::randu;

        elem_t epsilon = 0.12;
        theta1 = (randu<mat_t>(64, input.images.n_cols) * 2 - 1) * epsilon;
        theta2 = (randu<mat_t>(10, theta1.n_rows) * 2 - 1) * epsilon;
        theta1_bias = (randu<mat_t>(theta1.n_rows, 1) * 2 - 1) * epsilon;
        theta2_bias = (randu<mat_t>(theta2.n_rows, 1) * 2 - 1) * epsilon;
        d_theta1 = zeros<mat_t>(theta1.n_rows, theta1.n_cols);
        d_theta2 = zeros<mat_t>(theta2.n_rows, theta2.n_cols);
        d_theta1_bias = zeros<mat_t>(theta1.n_rows, 1);
        d_theta2_bias = zeros<mat_t>(theta2.n_rows, 1);
        // Convert the column vector of y labels to a matrix where each
        // row has a 1 in the column specified by the label.
        yy = zeros<mat_t>(input.labels.n_rows, 10);
        int row = 0;
        input.labels.for_each([&](const elem_t& element) {
            yy(row, element) = 1.0;
            row++;
        });

        // Allocate space for each layer of activations. Add an additional
        // column for the bias neuron
        a1 = ones<mat_t>(input.images.n_rows, input.images.n_cols);
        a2 = ones<mat_t>(input.images.n_rows, theta1.n_rows);
        a3 = ones<mat_t>(input.images.n_rows, theta2.n_rows);

        delta2 = zeros<mat_t>(a2.n_rows, a2.n_cols);
        delta3 = zeros<mat_t>(a3.n_rows, a3.n_cols);

        // Load the images onto first activation layer, but do not overwrite
        // the bias neuron
        a1 = input.images;

        bias = ones<mat_t>(a1.n_rows, 1);
    }


    /**
     * Use the neural network to predict outcomes
     *
     * @return The percentage of correct predictions

     */
    elem_t predict(void) {
        feed_forward();

        // Find neuron with maximum confidence, this is our predicted label
        arma::ucolvec predictions = index_max(a3, 1);

        // Display percentage of correct labels
        return elem_t(sum(predictions == input.labels)) / input.labels.n_rows;
    }

    /**
     * Cost function for the neural net
     *
     * @return the cost
     */
    elem_t cost() const {
        // Cost, without regularization
        // elem_t cost = sum(sum(
        //     -1 * yy % log(a3) - (1 - yy) % log(1 - a3), 1)) /
        //         input.images.n_rows;

        const elem_t * yy_ptr = yy.memptr();
        const elem_t * a3_ptr = a3.memptr();
        const size_t sz = yy.n_rows * yy.n_cols;

        elem_t cost = 0;
        for (size_t idx = 0; idx < yy.n_rows * yy.n_cols; ++idx) {
            elem_t yy_val = yy_ptr[idx];
            elem_t a3_val = a3_ptr[idx];
            cost += elem_t(-1) * yy_val * log(a3_val) -
                (1 - yy_val) * log(1 - a3_val);
        }
        cost /= input.images.n_rows;

        // Cost, with regularization
        if (std::abs(lambda) > 0) {
            // // Square each element. This next operation makes a copy
            // mat_t theta1_sq = square(theta1);
            // mat_t theta2_sq = square(theta2);
            //
            // // Sum up all elements in each layer
            // elem_t reg = (sum(sum(theta1_sq)) + sum(sum(theta2_sq)));
            //
            // // Normalize
            // reg *= lambda / (2 * input.images.n_rows);
            //
            // // Add in regularization term
            // cost += reg;

            const elem_t * theta1_ptr = theta1.memptr();
            const elem_t * theta2_ptr = theta2.memptr();
            elem_t reg = 0;
            for (size_t idx = 0; idx < theta1.n_rows * theta1.n_cols; ++idx) {
                reg += theta1_ptr[idx] * theta1_ptr[idx];
            }

            for (size_t idx = 0; idx < theta2.n_rows * theta2.n_cols; ++idx) {
                reg += theta2_ptr[idx] * theta2_ptr[idx];
            }

            reg *= lambda / (2 * input.images.n_rows);
            cost += reg;
        }

        return cost;
    }

    /**
     * Calculate the gradient for the weights
     */
    void gradient() {
        elem_t batch_size = a1.n_rows;

        // Activation delta from labeled data to current set of predictions
        // delta3 = a3 - yy;
        // rewrite as:
        // delta3 = -yy + a3
        // const size_t sz = delta3.n_rows * delta3.n_cols;
        // copy(sz, a3.memptr(), 1, delta3.memptr(), 1);
        // axpy(sz, elem_t(-1), yy.memptr(), 1, delta3.memptr(), 1);

        delta_labels(a3, yy, delta3);

        // Weight delta from output layer to hidden layer
        fc_gradient_weight(delta3, a2, bias, batch_size, d_theta2,
            d_theta2_bias);

        // Activation delta from output layer to hidden layer
        fc_gradient_activation(delta3, theta2, a2, delta2);

        // Weight delta from hidden layer to input layer
        fc_gradient_weight(delta2, a1, bias, batch_size, d_theta1,
            d_theta1_bias);

        // Regularization
        if (std::abs(lambda) > 0) {
            fc_regularize(theta1, theta1_bias, lambda, batch_size, d_theta1,
                d_theta1_bias);
            fc_regularize(theta2, theta2_bias, lambda, batch_size, d_theta2,
                d_theta2_bias);
        }
    }

    /**
     * A single Forward propogation step for the neural net
     */
    void feed_forward() {
        // Pass input thru weights to second layer
        fc_forward(a1, theta1, bias, theta1_bias, a2);
        sigmoid(a2);

        // Pass hidden layer to output layer
        fc_forward(a2, theta2, bias, theta2_bias, a3);
        sigmoid(a3);
    }

    /**
     * Train the neural network. Pass in a number of steps and a progress
     * function.
     *
     * \code
     * net.train(1000, [&](size_t i, size_t max_itr) -> void {
     *     if (i == 0 || i % 100 == 0 || i == max_itr - 1) {
     *         std::cout << "\r " << i << " j = " << train_net.cost()
     *                   << std::flush;
     *     }
     * });
     * \endcode
     *
     * @param steps    The number of steps to train
     * @param progress Function called after each step.
     */
    void train(size_t steps, std::function<void(size_t, size_t)> progress) {
        constexpr elem_t minus_one = elem_t(-1);
        for (size_t i = 0; i < steps; i++) {
            feed_forward();

            // Back prop
            gradient();

            // Apply gradient
            minus_eq(d_theta1, theta1);
            minus_eq(d_theta2, theta2);
            minus_eq(d_theta1_bias, theta1_bias);
            minus_eq(d_theta2_bias, theta2_bias);

            // // axpy(theta2.n_rows * theta2.n_cols, minus_one, d_theta2.memptr(),
            // //    0, theta2.memptr(), 0);
            // theta1_bias -= d_theta1_bias;
            // // axpy(theta1_bias.n_rows * theta1_bias.n_cols, minus_one,
            // //    d_theta1_bias.memptr(), 0,  theta1_bias.memptr(), 0);
            //
            // theta2_bias -= d_theta2_bias;
            // // axpy(theta2_bias.n_rows * theta2_bias.n_cols, minus_one,
            // //    d_theta2_bias.memptr(), 0,  theta2_bias.memptr(), 0);
            progress(i, steps);
        }
    }

    /**
     * Save the weights to two files in data_dir
     */
    void save() const {
        std::string theta1_fn = data_dir + "/theta1.bin";
        std::string theta2_fn = data_dir + "/theta2.bin";
        theta1.save(theta1_fn);
        theta2.save(theta2_fn);
    }
};

#endif /* end of include guard: NEURAL_NET_HPP_ */
