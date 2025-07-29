#pragma once

#include <Eigen/Dense>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

namespace DiFfRG
{
  /**
   * @brief Converts a dealii vector to an Eigen vector
   *
   * @param dealii a dealii vector
   * @param eigen an Eigen vector
   */
  void dealii_to_eigen(const dealii::Vector<double> &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts a dealii block vector to an Eigen vector
   *
   * @param dealii a dealii block vector
   * @param eigen an Eigen vector
   */
  void dealii_to_eigen(const dealii::BlockVector<double> &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts an Eigen vector to a dealii vector
   *
   * @param eigen an Eigen vector
   * @param dealii a dealii vector
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::Vector<double> &dealii);

  /**
   * @brief Converts an Eigen vector to a dealii block vector
   *
   * @param eigen an Eigen vector
   * @param dealii a dealii block vector
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::BlockVector<double> &dealii);
} // namespace DiFfRG