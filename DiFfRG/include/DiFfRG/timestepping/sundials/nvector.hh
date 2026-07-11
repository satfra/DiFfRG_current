#pragma once

#include <DiFfRG/timestepping/sundials/common.hh>

#include <sundials/sundials_nvector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace DiFfRG::sundials
{
  template <typename VectorType> N_Vector create_owned(std::unique_ptr<VectorType> vector, SUNContext context);

  template <typename VectorType> struct NVectorContent {
    explicit NVectorContent(VectorType *vector_) : vector(vector_) {}

    explicit NVectorContent(std::unique_ptr<VectorType> owned_)
        : owned(std::move(owned_)), vector(owned.get())
    {}

    std::unique_ptr<VectorType> owned;
    VectorType *vector = nullptr;
  };

  template <typename VectorType> VectorType &unwrap(N_Vector v)
  {
    return *static_cast<NVectorContent<VectorType> *>(v->content)->vector;
  }

  template <typename VectorType> const VectorType &unwrap_const(N_Vector v)
  {
    return *static_cast<NVectorContent<VectorType> *>(v->content)->vector;
  }

  namespace nvector_ops
  {
    template <typename VectorType> N_Vector_ID get_vector_id(N_Vector) { return SUNDIALS_NVEC_CUSTOM; }

    template <typename VectorType> N_Vector clone_empty(N_Vector w);

    template <typename VectorType> N_Vector clone(N_Vector w)
    {
      auto *source = static_cast<NVectorContent<VectorType> *>(w->content)->vector;
      auto owned = std::make_unique<VectorType>();
      owned->reinit(*source);
      return create_owned(std::move(owned), w->sunctx);
    }

    template <typename VectorType> void destroy(N_Vector v)
    {
      delete static_cast<NVectorContent<VectorType> *>(v->content);
      N_VFreeEmpty(v);
    }

    template <typename VectorType> void space(N_Vector, sunindextype *lrw, sunindextype *liw)
    {
      if (lrw != nullptr) *lrw = 0;
      if (liw != nullptr) *liw = 0;
    }

    template <typename VectorType> sunrealtype *get_array_pointer(N_Vector) { return nullptr; }
    template <typename VectorType> sunrealtype *get_device_array_pointer(N_Vector) { return nullptr; }
    template <typename VectorType> void set_array_pointer(sunrealtype *, N_Vector) {}
    template <typename VectorType> SUNComm get_communicator(N_Vector) { return SUN_COMM_NULL; }

    template <typename VectorType> sunindextype get_length(N_Vector v)
    {
      return static_cast<sunindextype>(unwrap_const<VectorType>(v).size());
    }

    template <typename VectorType> void linear_sum(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &yy = unwrap_const<VectorType>(y);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = a * xx[i] + b * yy[i];
    }

    template <typename VectorType> void constant(sunrealtype c, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = c;
    }

    template <typename VectorType> void product(N_Vector x, N_Vector y, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &yy = unwrap_const<VectorType>(y);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = xx[i] * yy[i];
    }

    template <typename VectorType> void divide(N_Vector x, N_Vector y, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &yy = unwrap_const<VectorType>(y);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = xx[i] / yy[i];
    }

    template <typename VectorType> void scale(sunrealtype c, N_Vector x, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = c * xx[i];
    }

    template <typename VectorType> void absolute(N_Vector x, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = std::abs(xx[i]);
    }

    template <typename VectorType> void inverse(N_Vector x, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = 1. / xx[i];
    }

    template <typename VectorType> void add_constant(N_Vector x, sunrealtype b, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = xx[i] + b;
    }

    template <typename VectorType> sunrealtype dot_product(N_Vector x, N_Vector y)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &yy = unwrap_const<VectorType>(y);
      sunrealtype sum = 0.;
      for (unsigned int i = 0; i < xx.size(); ++i) sum += xx[i] * yy[i];
      return sum;
    }

    template <typename VectorType> sunrealtype max_norm(N_Vector x)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      sunrealtype norm = 0.;
      for (unsigned int i = 0; i < xx.size(); ++i) norm = std::max<sunrealtype>(norm, std::abs(xx[i]));
      return norm;
    }

    template <typename VectorType> sunrealtype wrms_norm(N_Vector x, N_Vector w)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &ww = unwrap_const<VectorType>(w);
      sunrealtype sum = 0.;
      for (unsigned int i = 0; i < xx.size(); ++i) {
        const sunrealtype value = xx[i] * ww[i];
        sum += value * value;
      }
      return std::sqrt(sum / static_cast<sunrealtype>(xx.size()));
    }

    template <typename VectorType> sunrealtype wrms_norm_mask(N_Vector x, N_Vector w, N_Vector id)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &ww = unwrap_const<VectorType>(w);
      const auto &mask = unwrap_const<VectorType>(id);
      sunrealtype sum = 0.;
      for (unsigned int i = 0; i < xx.size(); ++i) {
        if (mask[i] > 0.) {
          const sunrealtype value = xx[i] * ww[i];
          sum += value * value;
        }
      }
      return std::sqrt(sum / static_cast<sunrealtype>(xx.size()));
    }

    template <typename VectorType> sunrealtype minimum(N_Vector x)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      sunrealtype value = std::numeric_limits<sunrealtype>::max();
      for (unsigned int i = 0; i < xx.size(); ++i) value = std::min<sunrealtype>(value, xx[i]);
      return value;
    }

    template <typename VectorType> sunrealtype wl2_norm(N_Vector x, N_Vector w)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      const auto &ww = unwrap_const<VectorType>(w);
      sunrealtype sum = 0.;
      for (unsigned int i = 0; i < xx.size(); ++i) {
        const sunrealtype value = xx[i] * ww[i];
        sum += value * value;
      }
      return std::sqrt(sum);
    }

    template <typename VectorType> sunrealtype l1_norm(N_Vector x)
    {
      const auto &xx = unwrap_const<VectorType>(x);
      sunrealtype sum = 0.;
      for (unsigned int i = 0; i < xx.size(); ++i) sum += std::abs(xx[i]);
      return sum;
    }

    template <typename VectorType> void compare(sunrealtype c, N_Vector x, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      for (unsigned int i = 0; i < zz.size(); ++i) zz[i] = (std::abs(xx[i]) >= c) ? 1. : 0.;
    }

    template <typename VectorType> sunbooleantype inv_test(N_Vector x, N_Vector z)
    {
      auto &zz = unwrap<VectorType>(z);
      const auto &xx = unwrap_const<VectorType>(x);
      bool ok = true;
      for (unsigned int i = 0; i < zz.size(); ++i) {
        if (xx[i] == 0.)
          ok = false;
        else
          zz[i] = 1. / xx[i];
      }
      return ok ? SUNTRUE : SUNFALSE;
    }

    template <typename VectorType> sunbooleantype constr_mask(N_Vector c, N_Vector x, N_Vector m)
    {
      const auto &cc = unwrap_const<VectorType>(c);
      const auto &xx = unwrap_const<VectorType>(x);
      auto &mm = unwrap<VectorType>(m);
      bool ok = true;
      for (unsigned int i = 0; i < xx.size(); ++i) {
        const bool failed = (cc[i] == 2. && xx[i] <= 0.) || (cc[i] == 1. && xx[i] < 0.) ||
                            (cc[i] == -1. && xx[i] > 0.) || (cc[i] == -2. && xx[i] >= 0.);
        mm[i] = failed ? 1. : 0.;
        ok = ok && !failed;
      }
      return ok ? SUNTRUE : SUNFALSE;
    }

    template <typename VectorType> sunrealtype min_quotient(N_Vector num, N_Vector denom)
    {
      const auto &nn = unwrap_const<VectorType>(num);
      const auto &dd = unwrap_const<VectorType>(denom);
      sunrealtype value = std::numeric_limits<sunrealtype>::max();
      bool found = false;
      for (unsigned int i = 0; i < nn.size(); ++i) {
        if (dd[i] != 0.) {
          value = std::min<sunrealtype>(value, nn[i] / dd[i]);
          found = true;
        }
      }
      return found ? value : std::numeric_limits<sunrealtype>::max();
    }
  } // namespace nvector_ops

  template <typename VectorType> N_Vector create_empty(SUNContext context)
  {
    N_Vector v = N_VNewEmpty(context);
    if (v == nullptr) throw std::bad_alloc();
    v->ops->nvgetvectorid = nvector_ops::get_vector_id<VectorType>;
    v->ops->nvclone = nvector_ops::clone<VectorType>;
    v->ops->nvcloneempty = nvector_ops::clone_empty<VectorType>;
    v->ops->nvdestroy = nvector_ops::destroy<VectorType>;
    v->ops->nvspace = nvector_ops::space<VectorType>;
    v->ops->nvgetarraypointer = nvector_ops::get_array_pointer<VectorType>;
    v->ops->nvgetdevicearraypointer = nvector_ops::get_device_array_pointer<VectorType>;
    v->ops->nvsetarraypointer = nvector_ops::set_array_pointer<VectorType>;
    v->ops->nvgetcommunicator = nvector_ops::get_communicator<VectorType>;
    v->ops->nvgetlength = nvector_ops::get_length<VectorType>;
    v->ops->nvgetlocallength = nvector_ops::get_length<VectorType>;
    v->ops->nvlinearsum = nvector_ops::linear_sum<VectorType>;
    v->ops->nvconst = nvector_ops::constant<VectorType>;
    v->ops->nvprod = nvector_ops::product<VectorType>;
    v->ops->nvdiv = nvector_ops::divide<VectorType>;
    v->ops->nvscale = nvector_ops::scale<VectorType>;
    v->ops->nvabs = nvector_ops::absolute<VectorType>;
    v->ops->nvinv = nvector_ops::inverse<VectorType>;
    v->ops->nvaddconst = nvector_ops::add_constant<VectorType>;
    v->ops->nvdotprod = nvector_ops::dot_product<VectorType>;
    v->ops->nvmaxnorm = nvector_ops::max_norm<VectorType>;
    v->ops->nvwrmsnorm = nvector_ops::wrms_norm<VectorType>;
    v->ops->nvwrmsnormmask = nvector_ops::wrms_norm_mask<VectorType>;
    v->ops->nvmin = nvector_ops::minimum<VectorType>;
    v->ops->nvwl2norm = nvector_ops::wl2_norm<VectorType>;
    v->ops->nvl1norm = nvector_ops::l1_norm<VectorType>;
    v->ops->nvcompare = nvector_ops::compare<VectorType>;
    v->ops->nvinvtest = nvector_ops::inv_test<VectorType>;
    v->ops->nvconstrmask = nvector_ops::constr_mask<VectorType>;
    v->ops->nvminquotient = nvector_ops::min_quotient<VectorType>;
    return v;
  }

  namespace nvector_ops
  {
    template <typename VectorType> N_Vector clone_empty(N_Vector w)
    {
      return create_empty<VectorType>(w->sunctx);
    }
  } // namespace nvector_ops

  class NVectorHandle
  {
  public:
    NVectorHandle() = default;
    explicit NVectorHandle(N_Vector vector_) : vector(vector_) {}
    NVectorHandle(const NVectorHandle &) = delete;
    NVectorHandle &operator=(const NVectorHandle &) = delete;

    NVectorHandle(NVectorHandle &&other) noexcept : vector(other.vector) { other.vector = nullptr; }

    NVectorHandle &operator=(NVectorHandle &&other) noexcept
    {
      if (this != &other) {
        reset();
        vector = other.vector;
        other.vector = nullptr;
      }
      return *this;
    }

    ~NVectorHandle() { reset(); }

    N_Vector get() const { return vector; }

    N_Vector release() noexcept
    {
      N_Vector released = vector;
      vector = nullptr;
      return released;
    }

    void reset(N_Vector replacement = nullptr)
    {
      if (vector != nullptr) N_VDestroy(vector);
      vector = replacement;
    }

  private:
    N_Vector vector = nullptr;
  };

  template <typename VectorType> N_Vector create_view(VectorType &vector, SUNContext context)
  {
    NVectorHandle guard(create_empty<VectorType>(context));
    auto content = std::make_unique<NVectorContent<VectorType>>(&vector);
    guard.get()->content = content.release();
    return guard.release();
  }

  template <typename VectorType> N_Vector create_owned(std::unique_ptr<VectorType> vector, SUNContext context)
  {
    NVectorHandle guard(create_empty<VectorType>(context));
    auto content = std::make_unique<NVectorContent<VectorType>>(std::move(vector));
    guard.get()->content = content.release();
    return guard.release();
  }
} // namespace DiFfRG::sundials
