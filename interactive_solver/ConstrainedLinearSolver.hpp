#pragma once

#define USE_CHOLMOD

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#ifdef USE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/Timer.h>

class LinearSolver {
public:
    LinearSolver() {}
    
    virtual void solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x) {};
};

class EigenCholesky : public LinearSolver {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol;
public:
    
    EigenCholesky(const Eigen::SparseMatrix<double>& A);

    void solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x);
};


#ifdef USE_CHOLMOD

class CholmodCholesky : public LinearSolver {
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> chol;
public:
    
    CholmodCholesky(const Eigen::SparseMatrix<double>& A);

    void solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x);
};

#endif

class ConstrainedLinearSolver {
    const double eps = 1e-8;
    const size_t n;

    Eigen::LDLT<Eigen::MatrixXd> B_inv;
    Eigen::MatrixXd solvedConstr;
    std::vector<size_t> constr;
    std::unique_ptr<LinearSolver> solver;

    igl::Timer t;
    int verbosity = 0;
public:

    void setVerbose(const int v = 1) {
        verbosity = v;
    }

    ConstrainedLinearSolver(const Eigen::SparseMatrix<double>& A) : n(A.cols()) {

        solvedConstr.resize(10, n);

        Eigen::SparseMatrix<double> Id(A.rows(), A.rows());
        Id.setIdentity();
        t.start();
        solver.reset(new EigenCholesky( (A + eps * Id)));
        t.stop();
        if (verbosity) std::cout << "initial factorization of a+eps id: " << t.getElapsedTimeInMilliSec() << " ms." << std::endl;
    }

    void updateBInv() {
        Eigen::MatrixXd  B(constr.size(), constr.size());
        for (int i = 0; i < constr.size(); ++i) {
            B.col(i) = solvedConstr.block(0, constr[i], constr.size(), 1);
        }

        B_inv = B.ldlt();
    }

    void addConstraint(int i) {
        if (constr.size() >= solvedConstr.rows()) {
            Eigen::MatrixXd solvedConstr2(2 * solvedConstr.rows(), n);
            solvedConstr2.topRows(solvedConstr.rows()) = solvedConstr;
            solvedConstr.swap(solvedConstr2);
        }

        t.start();
        constr.push_back(i);

        Eigen::MatrixXd xi;
        solver->solve(Eigen::VectorXd::Unit(n, i), xi);
        solvedConstr.row(constr.size() - 1) = xi.transpose();

        updateBInv();
        t.stop();
        if (verbosity) std::cout << "addConstraint: " << t.getElapsedTimeInMilliSec() << " ms." << std::endl;

    }

    void removeConstraint(int i) {

        t.start();

        auto it = std::find(constr.begin(), constr.end(), i);
        if (it == constr.end()) return;

        auto row = std::distance(constr.begin(), it);
        solvedConstr.middleRows(row, constr.size() - row - 1) = solvedConstr.middleRows(row + 1, constr.size() - row - 1).eval();
        constr.erase(it);

        updateBInv();
        t.stop();
        if (verbosity) std::cout << "removeConstraint: " << t.getElapsedTimeInMilliSec() << " ms." << std::endl;
    }


    void addConstraints(const std::vector<int>& ids) {
        for (int i : ids) addConstraint(i);
    }

    void removeConstraints(const std::vector<int>& ids) {
        for (int i : ids) removeConstraint(i);
    }

    void solve(Eigen::MatrixXd& b0, const Eigen::MatrixXd& handles, Eigen::MatrixXd& x) {

        assert(b0.rows() == x.rows() && b0.cols() == x.cols());

        Eigen::MatrixXd b = b0 + eps * x;
       
        t.start();
        if (constr.size()) {
            Eigen::MatrixXd b2 = handles - solvedConstr.topRows(constr.size()) * b;
            Eigen::MatrixXd x2 = B_inv.solve(b2);

 
            for (int i = 0; i < constr.size(); ++i) {
                b.row(constr[i]) += x2.row(i);
            }
        }
        t.stop();

        if (verbosity) std::cout << "solve 1 " << t.getElapsedTimeInMilliSec() << " ms\n";

        t.start();
        solver->solve(b, x);

        t.stop();
        if (verbosity) std::cout << "solve 2 " << t.getElapsedTimeInMilliSec() << " ms\n";
    }
};
