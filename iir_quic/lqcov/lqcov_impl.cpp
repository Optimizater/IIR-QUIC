#include "lqcov.hpp"
#include <iostream>

LqCov::LqCov(double lambda, double q, int max_iter, int max_time, double tol,
             double tol_ws, bool warm_start, bool msg)
    : lambda_(lambda), q_f(q), max_iter(max_iter), max_time(max_time), tol(tol),
      tol_ws(tol_ws), warm_start(warm_start), msg(msg)
{
}

double LqCov::lq_threshold(double z, double lambda_i, double q) const
{
    if (std::abs(q - 0.0) < NONZERO)
    {
        return z * (std::abs(z) > std::sqrt(2 * lambda_i) ? 1.0 : 0.0);
    }
    else if (std::abs(q - 1.0) < NONZERO)
    {
        return std::copysign(std::max(std::abs(z) - lambda_i, 0.0), z);
    }
    else
    {
        double beta_lambda = std::pow(2 * lambda_i * (1 - q), 1.0 / (2 - q));
        double h_lambda = beta_lambda + lambda_i * q * std::pow(beta_lambda, q - 1);

        if (std::abs(z) <= h_lambda)
        {
            return 0.0;
        }
        else
        {
            double beta = beta_lambda;
            for (int i = 0; i < 2000; i++) // Fixed-point iteration
            {
                double beta_new = std::abs(z) - lambda_i * q * std::pow(beta, q - 1);
                if (std::abs(beta_new - beta) < 1e-4)
                    break;
                beta = beta_new;
            }
            return std::copysign(beta, z);
        }
    }
}

VectorXd LqCov::update_u(const VectorXd &u, const MatrixXd &V_inv, const VectorXd &gamma,
                         double gamma_0, double q) const
{
    VectorXd u_hat = u;
    VectorXd u_hat_next = u;
    int len_u = u.size();

    for (int iter = 0; iter < 2000; iter++)
    {
        for (int i = 0; i < len_u; i++)
        {
            VectorXd u_temp = u_hat_next;
            u_temp(i) = 0.0;

            double denominator = gamma_0 * V_inv(i, i);
            // if (denominator < NONZERO)
            // {
            //     denominator = NONZERO;
            // }

            double z_i = -(gamma_0 * u_temp.dot(V_inv.col(i)) + gamma(i)) / denominator;
            double lambda_i = lambda_ / denominator;
            u_hat_next(i) = lq_threshold(z_i, lambda_i, q);
        }

        if ((u_hat - u_hat_next).norm() < 1e-4)
        {
            break;
        }

        u_hat = u_hat_next;
    }

    return u_hat_next;
}

MatrixXd LqCov::get_Omega_inv(const MatrixXd &V_inv, const VectorXd &u_plus,
                              double w_plus) const
{
    VectorXd z_plus = V_inv * u_plus;
    double c_plus = w_plus - u_plus.dot(z_plus);

    // if (c_plus < NONZERO)
    // {
    //     c_plus = NONZERO;
    // }

    MatrixXd Omega_plus_inv(V_inv.rows() + 1, V_inv.cols() + 1);
    Omega_plus_inv.topLeftCorner(V_inv.rows(), V_inv.cols()) =
        V_inv + z_plus * z_plus.transpose() / c_plus;
    Omega_plus_inv.topRightCorner(V_inv.rows(), 1) = -z_plus / c_plus;
    Omega_plus_inv.bottomLeftCorner(1, V_inv.cols()) = -z_plus.transpose() / c_plus;
    Omega_plus_inv(V_inv.rows(), V_inv.cols()) = 1.0 / c_plus;

    return Omega_plus_inv;
}

MatrixXd LqCov::get_V_inv(const MatrixXd &Omega_inv_perm) const
{
    int p = Omega_inv_perm.rows() - 1;
    MatrixXd P = Omega_inv_perm.topLeftCorner(p, p);
    VectorXd s = Omega_inv_perm.topRightCorner(p, 1);
    double r = Omega_inv_perm(p, p);

    double denominator = r;
    // if (denominator < NONZERO)
    // {
    //     denominator = NONZERO;
    // }

    return P - s * s.transpose() / denominator;
}

// -log(det(Omega)) + trace(S @ Omega) + lambda * sum(abs(Omega_d) ** q)
double LqCov::objective_function(const MatrixXd &Omega, const MatrixXd &S, double q) const
{
    Eigen::LLT<MatrixXd> llt(Omega);
    double log_det = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
    double trace = (Omega * S).trace();

    MatrixXd Omega_d = Omega;
    Omega_d.diagonal().setZero();
    double lq_term = lambda_ * Omega_d.array().abs().pow(q).sum();
    double L = log_det - trace - lq_term;

    return -L;
}

// -log(det(Omega)) + trace(S @ Omega) + lambda * sum(abs(Omega) ** q)
double LqCov::objective_function2(const MatrixXd &Omega, const MatrixXd &S,
                                  double q) const
{
    Eigen::LLT<MatrixXd> llt(Omega);
    double log_det = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
    double trace = (Omega * S).trace();

    double lq_term = lambda_ * Omega.array().abs().pow(q).sum();
    double L = log_det - trace - lq_term;

    return -L;
}

double LqCov::kkt_condition(const MatrixXd &Omega, const MatrixXd &Omega_inv,
                            const MatrixXd &S, double q, int p) const
{
    MatrixXd residual = S - Omega_inv;
    double optRes_unscaled = 0.0;

    for (int i = 0; i < p; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            if (std::abs(Omega(i, j)) > NONZERO)
            {
                if (i == j)
                {
                    double grad_ii = residual(i, j);
                    optRes_unscaled = std::max(optRes_unscaled, std::abs(grad_ii));
                }
                else
                {
                    double grad_ij =
                        residual(i, j) +
                        lambda_ * q *
                            std::copysign(std::pow(std::abs(Omega(i, j)), q - 1),
                                          Omega(i, j));
                    optRes_unscaled = std::max(optRes_unscaled, std::abs(grad_ij));
                }
            }
        }
    }

    double optRes = optRes_unscaled * p;

    return optRes;
}

double LqCov::kkt_condition2(const MatrixXd &Omega, const MatrixXd &Omega_inv,
                             const MatrixXd &S, double q, int p) const
{
    MatrixXd residual = S - Omega_inv;
    double optRes_unscaled = 0.0;

    for (int i = 0; i < p; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            if (std::abs(Omega(i, j)) > NONZERO)
            {
                double grad_ij =
                    residual(i, j) +
                    lambda_ * q *
                        std::copysign(std::pow(std::abs(Omega(i, j)), q - 1),
                                      Omega(i, j));
                optRes_unscaled = std::max(optRes_unscaled, std::abs(grad_ij));
            }
        }
    }

    double optRes = optRes_unscaled * p;

    return optRes;
}

MatrixXd LqCov::fit(const MatrixXd &S, const MatrixXd &Omega_ini)
{
    int p = S.rows();
    MatrixXd Omega = Omega_ini;
    MatrixXd Omega_inv = MatrixXd::Zero(p, p);

    f_val_list.clear();

    std::vector<double> q_values;
    if (warm_start)
    {
        int K = (q_f >= 0.9)   ? 2
                : (q_f >= 0.7) ? 3
                : (q_f >= 0.4) ? 4
                : (q_f >= 0.2) ? 5
                               : 6;

        q_values.resize(K);
        for (int i = 0; i < K; i++)
        {
            q_values[i] = 1.0 - (1.0 - q_f) * i / (K - 1);
        }
    }
    else
    {
        q_values.push_back(q_f);
    }

    for (double q_current : q_values)
    {
        if (std::abs(q_current - 1.0) < NONZERO)
        {
            // For q=1, we would need to implement graphical lasso here
            f_val_list.push_back(objective_function(Omega, S, q_current));
        }
        else
        {
            for (int cnt = 0; cnt < max_iter; cnt++)
            {
                int k = cnt % p;

                // Permutation indices
                std::vector<int> perm(p);
                for (int i = 0; i < p; i++)
                    perm[i] = i;
                std::swap(perm[k], perm[p - 1]);

                // Permute matrices
                MatrixXd S_perm(p, p);
                MatrixXd Omega_perm(p, p);
                MatrixXd Omega_inv_perm(p, p);

                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        S_perm(i, j) = S(perm[i], perm[j]);
                        Omega_perm(i, j) = Omega(perm[i], perm[j]);
                        Omega_inv_perm(i, j) = Omega_inv(perm[i], perm[j]);
                    }
                }

                // Extract submatrices
                MatrixXd V = Omega_perm.topLeftCorner(p - 1, p - 1);
                VectorXd u = Omega_perm.topRightCorner(p - 1, 1);

                VectorXd gamma = S_perm.topRightCorner(p - 1, 1);
                double gamma_0 = S_perm(p - 1, p - 1);

                MatrixXd V_inv;
                if (cnt == 0)
                {
                    V_inv = V.inverse();
                }
                else
                {
                    V_inv = get_V_inv(Omega_inv_perm);
                }

                // Update u and w
                VectorXd u_next = update_u(u, V_inv, gamma, gamma_0, q_current);
                double w_next = u_next.transpose() * V_inv * u_next + 1.0 / gamma_0;

                // if (w_next < NONZERO)
                // {
                //     w_next = NONZERO;
                // }

                // Update Omega_perm
                Omega_perm.topRightCorner(p - 1, 1) = u_next;
                Omega_perm.bottomLeftCorner(1, p - 1) = u_next.transpose();
                Omega_perm(p - 1, p - 1) = w_next;

                // Update Omega and Omega_inv
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Omega(perm[i], perm[j]) = Omega_perm(i, j);
                    }
                }

                Omega_inv_perm = get_Omega_inv(V_inv, u_next, w_next);
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Omega_inv(perm[i], perm[j]) = Omega_inv_perm(i, j);
                    }
                }

                // Store objective value
                f_val_list.push_back(objective_function(Omega, S, q_current));

                // Check convergence
                if (f_val_list.size() >= 3)
                {
                    double diff1 =
                        std::abs(f_val_list.back() - f_val_list[f_val_list.size() - 2]);
                    double diff2 = std::abs(f_val_list[f_val_list.size() - 2] -
                                            f_val_list[f_val_list.size() - 3]);

                    if (0.0 < diff1 && diff1 < tol && 0.0 < diff2 && diff2 < tol)
                    {
                        break;
                    }
                }
            }
        }
    }

    return Omega;
}

MatrixXd LqCov::fit2(const MatrixXd &S, const MatrixXd &Omega_ini)
{
    int p = S.rows();
    MatrixXd Omega = Omega_ini;
    MatrixXd Omega_inv = MatrixXd::Zero(p, p);

    f_val_list.clear();
    f_val_list_out.clear();
    KKT_list.clear();
    time_list.clear();

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> q_values;
    if (warm_start)
    {
        int K = (q_f >= 0.9)   ? 2
                : (q_f >= 0.7) ? 3
                : (q_f >= 0.4) ? 4
                : (q_f >= 0.2) ? 5
                               : 6;

        q_values.resize(K);
        for (int i = 0; i < K; i++)
        {
            q_values[i] = 1.0 - (1.0 - q_f) * i / (K - 1);
        }
    }
    else
    {
        q_values.push_back(q_f);
    }

    for (double q_current : q_values)
    {
        if (std::abs(q_current - 1.0) < NONZERO)
        {
            // For q=1, we would need to implement graphical lasso here
            f_val_list.push_back(objective_function(Omega, S, q_current));

            if (msg)
            {
                std::cout << "Finish graphical Lasso" << std::endl;
            }
        }
        else
        {
            for (int cnt = 0; cnt < max_iter; cnt++)
            {
                int k = cnt % p;

                if (msg)
                {
                    std::cout << "Iteration: " << cnt + 1 << " / " << max_iter
                              << ", q_current: " << q_current
                              << ", (== q_f: " << (std::abs(q_current - q_f) < NONZERO)
                              << ")" << std::endl;
                }

                // Permutation indices
                std::vector<int> perm(p);
                for (int i = 0; i < p; i++)
                    perm[i] = i;
                std::swap(perm[k], perm[p - 1]);

                // Permute matrices
                MatrixXd S_perm(p, p);
                MatrixXd Omega_perm(p, p);
                MatrixXd Omega_inv_perm(p, p);

                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        S_perm(i, j) = S(perm[i], perm[j]);
                        Omega_perm(i, j) = Omega(perm[i], perm[j]);
                        Omega_inv_perm(i, j) = Omega_inv(perm[i], perm[j]);
                    }
                }

                // Extract submatrices
                MatrixXd V = Omega_perm.topLeftCorner(p - 1, p - 1);
                VectorXd u = Omega_perm.topRightCorner(p - 1, 1);

                VectorXd gamma = S_perm.topRightCorner(p - 1, 1);
                double gamma_0 = S_perm(p - 1, p - 1);

                MatrixXd V_inv;
                if (cnt == 0)
                {
                    V_inv = V.inverse();
                }
                else
                {
                    V_inv = get_V_inv(Omega_inv_perm);
                }

                // Update u and w
                VectorXd u_next = update_u(u, V_inv, gamma, gamma_0, q_current);
                double w_next = u_next.transpose() * V_inv * u_next + 1.0 / gamma_0;

                // if (w_next < NONZERO)
                // {
                //     w_next = NONZERO;
                // }

                // Update Omega_perm
                Omega_perm.topRightCorner(p - 1, 1) = u_next;
                Omega_perm.bottomLeftCorner(1, p - 1) = u_next.transpose();
                Omega_perm(p - 1, p - 1) = w_next;

                // Update Omega and Omega_inv
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Omega(perm[i], perm[j]) = Omega_perm(i, j);
                    }
                }

                Omega_inv_perm = get_Omega_inv(V_inv, u_next, w_next);
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Omega_inv(perm[i], perm[j]) = Omega_inv_perm(i, j);
                    }
                }

                // Check convergence
                if (std::abs(q_current - q_f) < NONZERO)
                {
                    // double f_val = objective_function(Omega, S, q_f);  // CHANGE: fun2
                    double f_val = objective_function2(Omega, S, q_f);
                    f_val_list_out.push_back(f_val);

                    // MatrixXd residual = S - Omega_inv;
                    // double optRes_unscaled = 0.0;

                    // for (int i = 0; i < p; ++i)
                    // {
                    //     for (int j = 0; j < p; ++j)
                    //     {
                    //         if (std::abs(Omega(i, j)) > NONZERO)
                    //         {
                    //             if (i == j)
                    //             {
                    //                 double grad_ii = residual(i, j);
                    //                 optRes_unscaled =
                    //                     std::max(optRes_unscaled, std::abs(grad_ii));
                    //             }
                    //             else
                    //             {
                    //                 double grad_ij =
                    //                     residual(i, j) +
                    //                     lambda_ * q_f *
                    //                         std::copysign(
                    //                             std::pow(std::abs(Omega(i, j)), q_f - 1),
                    //                             Omega(i, j));
                    //                 optRes_unscaled =
                    //                     std::max(optRes_unscaled, std::abs(grad_ij));
                    //             }
                    //         }
                    //     }
                    // }

                    // double optRes = optRes_unscaled * p;

                    // KKT_list.push_back(optRes);

                    double optRes = kkt_condition(Omega, Omega_inv, S, q_f, p);

                    KKT_list.push_back(kkt_condition2(Omega, Omega_inv, S, q_f, p));  // CHANGE: kkt2

                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_time = end_time - start_time;
                    double elapsed_time_sec = elapsed_time.count();
                    time_list.push_back(elapsed_time_sec);

                    if (msg)
                    {
                        std::cout << "Objective value: " << f_val
                                  << ", KKT residual: " << optRes
                                  << ", elapsed Time: " << elapsed_time_sec << " seconds"
                                  << std::endl;
                    }

                    if (optRes < tol)
                    {
                        if (msg)
                        {
                            std::cout << "Converged with KKT residual: " << optRes
                                      << ", elapsed Time: " << elapsed_time_sec
                                      << " seconds" << std::endl;
                        }

                        break;
                    }
                }
                else
                {
                    f_val_list.push_back(objective_function(Omega, S, q_current));
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_time = end_time - start_time;

                    if (f_val_list.size() >= 3)
                    {
                        double diff1 = std::abs(f_val_list.back() -
                                                f_val_list[f_val_list.size() - 2]);
                        double diff2 = std::abs(f_val_list[f_val_list.size() - 2] -
                                                f_val_list[f_val_list.size() - 3]);
                        if (0.0 < diff1 && diff1 < tol_ws && 0.0 < diff2 &&
                            diff2 < tol_ws)
                        {
                            break;
                        }
                    }
                }
            }
        }
    }
    return Omega;
}

MatrixXd LqCov::fit3(const MatrixXd &S, const MatrixXd &Omega_ini)
{
    int p = S.rows();
    MatrixXd Omega = Omega_ini;
    MatrixXd Omega_inv = MatrixXd::Zero(p, p);

    f_val_list.clear();
    f_val_list_out.clear();
    KKT_list.clear();
    time_list.clear();

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> q_values;
    if (warm_start)
    {
        int K = (q_f >= 0.9)   ? 2
                : (q_f >= 0.7) ? 3
                : (q_f >= 0.4) ? 4
                : (q_f >= 0.2) ? 5
                               : 6;

        q_values.resize(K);
        for (int i = 0; i < K; i++)
        {
            q_values[i] = 1.0 - (1.0 - q_f) * i / (K - 1);
        }
    }
    else
    {
        q_values.push_back(q_f);
    }

    for (double q_current : q_values)
    {
        if (std::abs(q_current - 1.0) < NONZERO)
        {
            // For q=1, we would need to implement graphical lasso here
            f_val_list.push_back(objective_function(Omega, S, q_current));

            if (msg)
            {
                std::cout << "Finish graphical Lasso" << std::endl;
            }
        }
        else
        {
            int cnt = 0;

            while (true)
            {
                int k = cnt % p;

                if (msg)
                {
                    std::cout << "Iteration: " << cnt + 1 << " / " << max_iter
                              << ", q_current: " << q_current
                              << ", (== q_f: " << (std::abs(q_current - q_f) < NONZERO)
                              << ")" << std::endl;
                }

                // Permutation indices
                std::vector<int> perm(p);
                for (int i = 0; i < p; i++)
                    perm[i] = i;
                std::swap(perm[k], perm[p - 1]);

                // Permute matrices
                MatrixXd S_perm(p, p);
                MatrixXd Omega_perm(p, p);
                MatrixXd Omega_inv_perm(p, p);

                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        S_perm(i, j) = S(perm[i], perm[j]);
                        Omega_perm(i, j) = Omega(perm[i], perm[j]);
                        Omega_inv_perm(i, j) = Omega_inv(perm[i], perm[j]);
                    }
                }

                // Extract submatrices
                MatrixXd V = Omega_perm.topLeftCorner(p - 1, p - 1);
                VectorXd u = Omega_perm.topRightCorner(p - 1, 1);

                VectorXd gamma = S_perm.topRightCorner(p - 1, 1);
                double gamma_0 = S_perm(p - 1, p - 1);

                MatrixXd V_inv;
                if (cnt == 0)
                {
                    V_inv = V.inverse();
                }
                else
                {
                    V_inv = get_V_inv(Omega_inv_perm);
                }

                cnt++;

                // Update u and w
                VectorXd u_next = update_u(u, V_inv, gamma, gamma_0, q_current);
                double w_next = u_next.transpose() * V_inv * u_next + 1.0 / gamma_0;

                // if (w_next < NONZERO)
                // {
                //     w_next = NONZERO;
                // }

                // Update Omega_perm
                Omega_perm.topRightCorner(p - 1, 1) = u_next;
                Omega_perm.bottomLeftCorner(1, p - 1) = u_next.transpose();
                Omega_perm(p - 1, p - 1) = w_next;

                // Update Omega and Omega_inv
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Omega(perm[i], perm[j]) = Omega_perm(i, j);
                    }
                }

                Omega_inv_perm = get_Omega_inv(V_inv, u_next, w_next);
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Omega_inv(perm[i], perm[j]) = Omega_inv_perm(i, j);
                    }
                }

                // Check convergence
                if (std::abs(q_current - q_f) < NONZERO)
                {
                    // double f_val = objective_function(Omega, S, q_f);  // CHANGE: fun2
                    double f_val = objective_function2(Omega, S, q_f);
                    f_val_list_out.push_back(f_val);

                    // MatrixXd residual = S - Omega_inv;
                    // double optRes_unscaled = 0.0;

                    // for (int i = 0; i < p; ++i)
                    // {
                    //     for (int j = 0; j < p; ++j)
                    //     {
                    //         if (std::abs(Omega(i, j)) > NONZERO)
                    //         {
                    //             if (i == j)
                    //             {
                    //                 double grad_ii = residual(i, j);
                    //                 optRes_unscaled =
                    //                     std::max(optRes_unscaled, std::abs(grad_ii));
                    //             }
                    //             else
                    //             {
                    //                 double grad_ij =
                    //                     residual(i, j) +
                    //                     lambda_ * q_f *
                    //                         std::copysign(
                    //                             std::pow(std::abs(Omega(i, j)), q_f - 1),
                    //                             Omega(i, j));
                    //                 optRes_unscaled =
                    //                     std::max(optRes_unscaled, std::abs(grad_ij));
                    //             }
                    //         }
                    //     }
                    // }

                    // double optRes = optRes_unscaled * p;

                    // KKT_list.push_back(optRes);

                    double optRes = kkt_condition(Omega, Omega_inv, S, q_f, p);

                    KKT_list.push_back(kkt_condition2(Omega, Omega_inv, S, q_f, p));  // CHANGE: kkt2

                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_time = end_time - start_time;
                    double elapsed_time_sec = elapsed_time.count();
                    time_list.push_back(elapsed_time_sec);

                    if (msg)
                    {
                        std::cout << "Objective value: " << f_val
                                  << ", KKT residual: " << optRes
                                  << ", elapsed Time: " << elapsed_time_sec << " seconds"
                                  << std::endl;
                    }

                    if (optRes < tol || elapsed_time_sec >= max_time)
                    {
                        if (msg)
                        {
                            std::cout << "Converged with KKT residual: " << optRes
                                      << ", elapsed Time: " << elapsed_time_sec
                                      << " seconds" << std::endl;
                        }

                        break;
                    }
                }
                else
                {
                    f_val_list.push_back(objective_function(Omega, S, q_current));
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_time = end_time - start_time;

                    if (f_val_list.size() >= 3)
                    {
                        double diff1 = std::abs(f_val_list.back() -
                                                f_val_list[f_val_list.size() - 2]);
                        double diff2 = std::abs(f_val_list[f_val_list.size() - 2] -
                                                f_val_list[f_val_list.size() - 3]);
                        if ((0.0 < diff1 && diff1 < tol_ws && 0.0 < diff2 &&
                             diff2 < tol_ws) ||
                            cnt >= max_iter)
                        {
                            break;
                        }
                    }
                }
            }
        }
    }
    return Omega;
}
