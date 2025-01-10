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


    Eigen::MatrixXd xB;
    Eigen::VectorXi I, B;
    Eigen::MatrixXd b2;
    Eigen::Vector3i R3;

    const size_t n;
    
    Eigen::LDLT<Eigen::MatrixXd> B_inv;
    Eigen::MatrixXd solvedConstr;
    std::vector<size_t> constr;
    std::unique_ptr<LinearSolver> solver;
    
    igl::Timer t;
    int verbosity = 1;
public:
    
    void setVerbose(const int v = 1) {
        verbosity = v;
    }
    
    //builds system
    void init(const Eigen::SparseMatrix<double>& A,
              const std::vector<int>& constrFixed,
              const std::vector<Eigen::VectorXd>& values) {
        
        
        R3 << 0, 1, 2;
        solvedConstr.resize(10, n);
        
        
        // sort constraints
        std::vector<int> range;
        for (int i = 0; i < constrFixed.size(); ++i) range.push_back(i);
        std::sort(range.begin(), range.end(), [&](const int i, const int j) {return constrFixed[i] < constrFixed[j]; });

        // fill constraint values, index free constraints
        if (!values.empty()) {
            xB.resize(values.size(), values[0].size());
            B.resize(constrFixed.size());
        }

        int cnt = 0;
        for (int i : range) {
            B(cnt) = constrFixed[i];
            xB.row(cnt) = values[i];
            ++cnt;
        }

        std::vector<int> rangeFull;
        for (int i = 0; i < A.cols(); ++i) rangeFull.push_back(i);
        I.resize(A.cols() - constrFixed.size());
        std::set_difference(rangeFull.begin(), rangeFull.end(), B.data(), B.data() + B.size(), I.data());

        // build part matrices
        Eigen::SparseMatrix<double> AII, AIB;

        igl::slice(A, I, I, AII);
        
        t.start();
        solver.reset(new CholmodCholesky(AII));
        t.stop();
        if(verbosity) std::cout << "factorization of A: " << t.getElapsedTimeInMilliSec() << " ms." << std::endl;
        
        igl::slice(A, I, B, AIB);
        b2 = AIB * xB;
    }
    
    
    ConstrainedLinearSolver(const Eigen::SparseMatrix<double>& A,
                            const std::vector<int>& constrFixed,
                            const std::vector<Eigen::VectorXd>& values) : n(A.cols()) {
        
        init(A, constrFixed, values);
    }
    
    ConstrainedLinearSolver(const Eigen::SparseMatrix<double>& A) : n(A.cols()) {
        init(A, {}, {});
    }

    void updateBInv() {
        Eigen::MatrixXd  B(constr.size(), constr.size());
        for(int i = 0; i < constr.size(); ++i) {
            B.col(i) = solvedConstr.block(0, constr[i], constr.size(), 1);
        }
        
        //std::cout << B << std::endl;
        //B_inv = B.fullPivHouseholderQr();
        B_inv = B.ldlt();
    }
    
    void addConstraint(int i) {
        std::cout<<"add constr"<<std::endl;
        if(constr.size() >= solvedConstr.cols()) return;
        
        t.start();
        constr.push_back(i);
        std::cout<<"push"<<std::endl;
        // todo: directly place result?
        Eigen::MatrixXd xi;
        solve(Eigen::VectorXd::Unit(n, i), xi);
        std::cout<<"solve"<<std::endl;
        solvedConstr.row(constr.size() - 1) = xi.transpose();
        std::cout<<"thingy"<<std::endl;
        
        updateBInv();
        std::cout<<"update"<<std::endl;
        t.stop();
        if(verbosity) std::cout << "addConstraint (solving for Q again?): " << t.getElapsedTimeInMilliSec() << " ms." << std::endl;
               
    }
    
    void removeConstraint(int i) {
        
        t.start();
        
        auto it = std::find(constr.begin(), constr.end(), i);
        if(it == constr.end()) return;
        
        auto row = std::distance(constr.begin(), it);
        solvedConstr.middleRows(row, constr.size() - row - 1) = solvedConstr.middleRows(row + 1, constr.size() - row - 1).eval();
        constr.erase(it);
        
        updateBInv();
        t.stop();
        if(verbosity) std::cout << "removeConstraint: (solving for Q again?)" << t.getElapsedTimeInMilliSec() << " ms." << std::endl;
    }
    
    
    void addConstraints(const std::vector<int>& ids) {
        // todo: efficient implementation
        for(int i : ids) addConstraint(i);
    }
    
    void removeConstraints(const std::vector<int>& ids) {
        // todo: efficient implementation
        for(int i : ids) removeConstraint(i);
    }
    
    void solve( Eigen::MatrixXd& b0, const Eigen::MatrixXd& handles, Eigen::MatrixXd& x) {
    
        t.start();
        if(constr.size()) {
            Eigen::MatrixXd b2 = handles - solvedConstr.topRows(constr.size()) * b0;
            Eigen::MatrixXd x2 = B_inv.solve(b2);
            
            for(int i = 0; i < constr.size(); ++i) {
                b0.row(constr[i]) += x2.row(i);
            }
        }
        t.stop();

        if(verbosity) std::cout << "solve 1 (Lagrangian dense Q system)" <<  t.getElapsedTimeInMilliSec() << " ms\n";
        
        t.start();
        solve(b0, x);
        
        t.stop();
        if(verbosity) std::cout << "solve 2 (vertex positions)" <<  t.getElapsedTimeInMilliSec() << " ms\n";
    }
    
    //solves system for given rhs b
    void solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x) {
        std::cout<<b.rows()<<b.cols()<<std::endl;
        if(b2.size() > 0) { // are fixed constraints present
            std::cout<<"hiya"<<std::endl;
            int C_size=b.cols();
            Eigen::VectorXi C(C_size);
            for(int i = 0; i < C_size; ++i) C(i) = i;
            std::cout<<C<<std::endl;
            std::cout<<"slice is nice"<<std::endl;
            Eigen::MatrixXd bI;
            igl::slice(b, I, C, bI);
            std::cout<<"hmm"<<std::endl;
            Eigen::MatrixXd xI;
            std::cout<<bI.rows()<<std::endl;
            std::cout<<bI.cols()<<std::endl;
            std::cout<<b2.rows()<<std::endl;
            std::cout<<b2.cols()<<std::endl;
            solver->solve(bI - b2, xI);
            std::cout<<"yeahh"<<std::endl;
            x.resizeLike(b);
            std::cout<<"whats"<<std::endl;
            std::cout<<xB<<std::endl;
            std::cout<<B<<std::endl;
            std::cout<<C<<std::endl;
            //std::cout<<x<<std::endl;
            std::cout<<"heheh"<<x.rows()<<" "<<x.cols()<<" "<<x.norm()<<std::endl;
            igl::slice_into(xB, B, C, x);
            std::cout<<"uppp"<<std::endl;
            igl::slice_into(xI, I, C, x);
            std::cout<<"let's gooo"<<std::endl;
        } else {
            solver->solve(b, x);
        }
    }
};

