#ifndef FC_HPP_
#define FC_HPP_

/** \file fc.hpp
 * \brief fully connected layer functions
 */

#include <armadillo>

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
 * Calculate the gradient for the activations between two layers,
 * layer 1 activations * theta -> layer 2 activations
 *
 * @param delta       delta for layer 2 activations
 *                    (batch_size x output features)
 * @param theta       weights for layer 1
 *                    (output features x input features)
 * @param activation  activations for layer 1
 *                    (batch size x input features)
 * @param result      delta for layer 1 activations
 *                    (batch size x input features)
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

#endif /* end of include guard: FC_HPP_ */
