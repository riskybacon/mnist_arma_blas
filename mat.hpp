#ifndef MAT_HPP_
#define MAT_HPP_

#include <fstream>
#include <random>

template<typename Mat> Mat zeros(size_t n_rows, size_t n_cols);
template<typename Mat> Mat ones(size_t n_rows, size_t n_cols);
template<typename Mat> Mat randu(size_t n_rows, size_t n_cols);

template<typename T>
struct mat {
    using elem_t = T;
    using elem_type = T;

    size_t n_rows;
    size_t n_cols;

    elem_t * ptr = nullptr;

    mat(size_t n_rows_ = 0, size_t n_cols_ = 0)
    : n_rows(n_rows_),
      n_cols(n_cols_) {
        if (n_rows > 0 && n_cols > 0) {
            // TODO: fix memory leak. Don't just put delete ptr in destructor,
            // you need to handle copy construction and assignment properly
            ptr = new elem_t[n_rows_ * n_cols_];
        }
    }

    mat(const mat& other)
    : n_rows(other.n_rows),
      n_cols(other.n_cols) {
        if (n_rows > 0 && n_cols > 0) {
            ptr = new elem_t[n_rows * n_cols];
            memcpy(ptr, other.ptr, n_rows * n_cols * sizeof(elem_t));
        }
    }

    mat& operator=(const mat& other) {
        if (this != &other) {
            if (n_rows != other.n_rows || n_cols != other.n_cols) {
                if (ptr != nullptr) {
                    delete ptr;
                }
                n_rows = other.n_rows;
                n_cols = other.n_cols;
                ptr = new elem_t[n_rows * n_cols];
            }
            memcpy(ptr, other.ptr, n_rows * n_cols * sizeof(elem_t));
        }
        return *this;
    }

    ~mat() {
        if (ptr != nullptr) {
            delete ptr;
            ptr = nullptr;
        }
    }

    const elem_t* memptr() const {
        return ptr;
    }

    elem_t* memptr() {
        return ptr;
    }

    elem_t& operator()(size_t row, size_t col) {
        return ptr[col * n_rows + row];
    }

    const elem_t& operator()(size_t row, size_t col) const {
        return ptr[col * n_rows + row];
    }

    elem_t& operator[](size_t idx) {
        return ptr[idx];
    }

    const elem_t& operator[](size_t idx) const {
        return ptr[idx];
    }

    void transform(std::function<elem_t(elem_t)> map) {
        for (size_t idx = 0; idx < n_rows * n_cols; ++idx) {
            ptr[idx] = map(ptr[idx]);
        }
    }

    void for_each(std::function<void(elem_t)> map) const {
        for (size_t idx = 0; idx < n_rows * n_cols; ++idx) {
            map(ptr[idx]);
        }
    }

    elem_t min() const {
        elem_t min = ptr[0];
        for_each([&](elem_t val) {
            if (val < min) {
                min = val;
            }
        });

        return min;
    }

    elem_t max() const {
        elem_t max = ptr[0];
        for_each([&](elem_t val) {
            if (val > max) {
                max = val;
            }
        });

        return max;
    }

    mat operator==(const mat& other) const {
        mat equal = zeros<mat>(n_rows, n_cols);

        for (size_t idx = 0; idx < n_rows * n_cols; ++idx) {
            equal[idx] = (ptr[idx] == other[idx]);
        }

        return equal;
    }
};

template<typename Mat>
Mat constant(size_t n_rows, size_t n_cols, typename Mat::elem_t val) {
    using elem_t = typename Mat::elem_type;
    Mat retval(n_rows, n_cols);
    retval.transform([&](elem_t) {
        return val;
    });
    return retval;
}

/**
 * Create a matrix with all elements initialized to zero
 *
 * @param n_rows  number of rows in the matrix
 * @param n_cols  number of columns in the matrix
 *
 * @return A n_rows x n_cols matrix with all elements set to zero
 */
template<typename Mat>
Mat zeros(size_t n_rows, size_t n_cols) {
    return constant<Mat>(n_rows, n_cols, typename Mat::elem_t(0));
}

/**
 * Create a matrix with all elements initialized to one
 *
 * @param n_rows  number of rows in the matrix
 * @param n_cols  number of columns in the matrix
 *
 * @return A n_rows x n_cols matrix with all elements set to one
 */
template<typename Mat>
Mat ones(size_t n_rows, size_t n_cols) {
    return constant<Mat>(n_rows, n_cols, typename Mat::elem_t(1));
}

/**
 * Create a matrix with all elements initialized to uniform random numbers
 *
 * @param n_rows  number of rows in the matrix
 * @param n_cols  number of columns in the matrix
 * @param min     minimum value in the distribution
 * @param max     maximum value in the distribution
 *
 * @return A n_rows x n_cols matrix with all elements set to random values
 *    drawn from a uniform distribution
 */
template<typename Mat>
Mat randu(size_t n_rows, size_t n_cols, typename Mat::elem_t min,
    typename Mat::elem_t max
) {
    using elem_t = typename Mat::elem_t;
    Mat retval(n_rows, n_cols);
    std::default_random_engine generator;
    std::uniform_real_distribution<elem_t> distribution(min, max);

    retval.transform([&](elem_t) {
        return distribution(generator);
    });

    return retval;
}

/**
 * For each row in a matrix, find the index of the column with the greatest
 * value.
 *
 * @param mt  The matrix (n_rows x n_cols)
 * @return a matrix (n_rows x 1) with the index of the column with the greatest
 *     value
 */
template<
    typename T,
    typename std::enable_if<
        std::is_arithmetic<T>::value,
        int
    >::type = 0
>
mat<unsigned int> col_max(const mat<T>& mt) {
    using elem_t = T;
    mat<unsigned int> retval(mt.n_rows, 1);

    for (size_t row = 0; row < mt.n_rows; ++row) {
        elem_t max_val = mt(row, 0);
        size_t max_col = 0;
        for (size_t col = 1; col < mt.n_cols; ++col) {
            elem_t val = mt(row, col);
            if (val > max_val) {
                max_val = val;
                max_col = col;
            }
        }

        retval(row, 0) = max_col;
    }
    return retval;
}

/**
 * Sum all elements in a matrix
 *
 * @param matrix  The matrix to sum
 * @return The sum
 */
template<typename T>
T sum(const mat<T>& matrix) {
    T result = 0;
    matrix.for_each([&](const T& elem) {
        result += elem;
    });
    return result;
}

// template<typename T>
// std::ostream& operator<<(std::ostream& out, const mat<T>& mt) {
//     for (size_t row = 0; row < mt.n_rows; ++row) {
//         std::cout << "row: " << row << std::endl;
//         for (size_t col = 0; col < mt.n_cols; ++col) {
//             T val = mt(row, col);
//             // out << col << ": " << val << ", ";
//         }
//         out << std::endl;
//     }
//     return out;
// }
#endif /* end of include guard: MAT_HPP_ */
