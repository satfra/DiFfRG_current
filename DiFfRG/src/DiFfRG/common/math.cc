// DiFfRG
#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  void dealii_to_eigen(const dealii::Vector<double> &dealii, Eigen::VectorXd &eigen)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) eigen.resize(dealii.size());
    for (uint i = 0; i < dealii.size(); ++i)
      eigen(i) = dealii(i);
  }

  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::Vector<double> &dealii)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) dealii.reinit(eigen.size());
    for (uint i = 0; i < eigen.size(); ++i)
      dealii(i) = eigen(i);
  }

  void dealii_to_eigen(const dealii::BlockVector<double> &dealii, Eigen::VectorXd &eigen)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) eigen.resize(dealii.size());
    for (uint i = 0; i < dealii.size(); ++i)
      eigen(i) = dealii(i);
  }

  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::BlockVector<double> &dealii)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size()))
      throw std::runtime_error("eigen_to_dealii: dealii and eigen vectors have different sizes!");
    for (uint i = 0; i < eigen.size(); ++i)
      dealii(i) = eigen(i);
  }
} // namespace DiFfRG