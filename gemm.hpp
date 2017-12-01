#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <Accelerate/Accelerate.h>

enum Transpose {
    Trans,
    NoTrans
};

#define convert_transpose(dst, src) \
    constexpr enum CBLAS_TRANSPOSE dst = src == Trans ? \
        CblasTrans : CblasNoTrans

enum ColumnMajor {
    ColMajor,
    RowMajor
};

#define convert_order(dst, src) \
    constexpr enum CBLAS_ORDER dst = src == ColMajor ? \
        CblasColMajor : CblasRowMajor

template<
    Transpose TP1,
    Transpose TP2,
    typename T,
    ColumnMajor Order = ColMajor,
    typename std::enable_if<
        std::is_same<T, float>::value,
        int
    >::type = 0
>
void gemm(
    const int m, const int n, const int k,
    const T alpha,
    const T *a, const int lda,
    const T *b, const int ldb,
    const T beta,
    T *c, const int ldc
) {
    convert_transpose(tp1, TP1);
    convert_transpose(tp2, TP2);
    convert_order(order, Order);
    cblas_sgemm(
        order, tp1, tp2, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
}

template<
    Transpose TP1,
    Transpose TP2,
    typename T,
    ColumnMajor Order = ColMajor,
    typename std::enable_if<
        std::is_same<T, double>::value,
        int
    >::type = 0
>
void gemm(
    const int m, const int n, const int k,
    const T alpha,
    const T *a,
    const int lda,
    const T *b,
    const int ldb,
    const T beta,
    T *c,
    const int ldc
) {
    convert_transpose(tp1, TP1);
    convert_transpose(tp2, TP2);
    convert_order(order, Order);
    cblas_dgemm(
        order, tp1, tp2, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
}

void axpy(int n, const float& sa, const float * const sx, const int incx,
    float * const sy, const int incy
) {
    cblas_saxpy(n, sa, sx, incx, sy, incy);
}

void axpy(int n, const double& sa, const double * const sx, const int incx,
    double * const sy, const int incy
) {
    cblas_daxpy(n, sa, sx, incx, sy, incy);
}

void scal(int n, const float& sa, float * const sx, const int incx) {
    cblas_sscal(n, sa, sx, incx);
}

void scal(int n, const double& sa, double * const sx, const int incx) {
    cblas_dscal(n, sa, sx, incx);
}

void copy(int n, const float * const sx, const int incx, float * const sy,
    const int incy
) {
    cblas_scopy(n, sx, incx, sy, incy);
}

void copy(int n, const double * const sx, const int incx, double * const sy,
    const int incy
) {
    cblas_dcopy(n, sx, incx, sy, incy);
}

template<typename T>
void minus_eq(const T& src, T& dst) {
    using elem_t = typename T::elem_type;
    // dst -= src
    // dst = -1 * src + dst;
    // y = a * x + y
    axpy(src.n_rows * src.n_cols, elem_t(-1), src.memptr(), 1, dst.memptr(), 1);
}

#endif /* end of include guard: GEMM_HPP_ */
