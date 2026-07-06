#ifndef LQCOV_HPP
#define LQCOV_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;

class LqCov {
public:
    LqCov(double lambda, double q, int max_iter, int max_time, double tol, double tol_ws, bool warm_start, bool msg);
    
    MatrixXd fit(const MatrixXd& S, const MatrixXd& Omega_ini);
    MatrixXd fit2(const MatrixXd& S, const MatrixXd& Omega_ini);
    MatrixXd fit3(const MatrixXd& S, const MatrixXd& Omega_ini);
    
    const std::vector<double>& get_fval_list() const { return f_val_list; }
    const std::vector<double>& get_fval_list_out() const { return f_val_list_out; }
    const std::vector<double>& get_KKT_list() const { return KKT_list; }
    const std::vector<double>& get_time_list() const { return time_list; }
    
private:
    double lambda_;
    double q_f;
    int max_iter;
    int max_time;
    double tol;
    double tol_ws;
    bool warm_start;
    bool msg;
    std::vector<double> f_val_list;
    std::vector<double> f_val_list_out;
    std::vector<double> KKT_list;
    std::vector<double> time_list;
    
    static constexpr double NONZERO = 1e-8;
    
    double lq_threshold(double z, double lambda_i, double q) const;
    VectorXd update_u(const VectorXd& u, const MatrixXd& V_inv, 
                     const VectorXd& gamma, double gamma_0, double q) const;
    MatrixXd get_Omega_inv(const MatrixXd& V_inv, const VectorXd& u_plus, double w_plus) const;
    MatrixXd get_V_inv(const MatrixXd& Omega_inv_perm) const;
    double objective_function(const MatrixXd& Omega, const MatrixXd& S, double q) const;
    double objective_function2(const MatrixXd& Omega, const MatrixXd& S, double q) const;
    double kkt_condition(const MatrixXd& Omega, const MatrixXd& Omega_inv, const MatrixXd& S, double q, int p) const;
    double kkt_condition2(const MatrixXd& Omega, const MatrixXd& Omega_inv, const MatrixXd& S, double q, int p) const;
};

#endif // LQCOV_HPP