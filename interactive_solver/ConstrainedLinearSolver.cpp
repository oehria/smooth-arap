#include "ConstrainedLinearSolver.hpp"


EigenCholesky::EigenCholesky(const Eigen::SparseMatrix<double>& A) {
    chol.compute(A);
}

void EigenCholesky::solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x) {
    x = chol.solve(b);
}

#ifdef USE_CHOLMOD

CholmodCholesky::CholmodCholesky(const Eigen::SparseMatrix<double>& A) {
    chol.compute(A);
}

void CholmodCholesky::solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x) {
    x = chol.solve(b);
}

#endif
