#pragma once

// standard library
#include <cmath>
#include <vector>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  /**
   * @brief A class representing a polynomial.
   */
  class Polynomial
  {
  public:
    /**
     * @brief Construct a new Polynomial object from a list of coefficients.
     * The coefficients are ordered from the lowest to the highest power, starting with 0.
     */
    Polynomial(const std::vector<double> &coefficients);

    /**
     * @brief Access the i-th coefficient of the polynomial.
     */
    const double &operator[](const uint &i) const;

    /**
     * @brief Access the i-th coefficient of the polynomial.
     */
    double &operator[](const uint &i);

    /**
     * @brief Evaluate the polynomial at a given point x.
     */
    double operator()(const double &x) const;

    /**
     * @brief Multiply the polynomial by a scalar.
     */
    void operator*=(const double &scalar);

    /**
     * @brief Multiply the polynomial by another polynomial.
     */
    void operator*=(const Polynomial &other);

    /**
     * @brief Compute the antiderivative of the polynomial. An integration constant is given as an argument.
     *
     * The antiderivative is a polynomial of one degree higher than the original polynomial.
     *
     * @param integration_constant The constant term of the antiderivative.
     * @return Polynomial The antiderivative of the polynomial.
     */
    Polynomial anti_derivative(const double &integration_constant) const;

    /**
     * @brief Compute the derivative of the polynomial.
     *
     * The derivative is a polynomial of one degree lower than the original polynomial.
     *
     * @return Polynomial The derivative of the polynomial.
     */
    Polynomial derivative() const;

    /**
     * @brief Compute the integral of the polynomial between two points.
     *
     * @param x_min The lower bound of the integral.
     * @param x_max The upper bound of the integral.
     * @return double The integral of the polynomial between x_min and x_max.
     */
    double integral(const double &x_min, const double &x_max) const;

    /**
     * @brief A utility function to obtain from this polynomial P(x) the polynomial Q(x) = P(x^2)
     *
     * @return Polynomial The polynomial Q(x) = P(x^2).
     */
    Polynomial square_argument() const;

  private:
    std::vector<double> coefficients;
  };
} // namespace DiFfRG
