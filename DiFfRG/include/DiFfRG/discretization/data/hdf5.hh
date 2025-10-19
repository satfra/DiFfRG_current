#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <autodiff/forward/real/real.hpp>

#ifdef H5CPP
#include <h5cpp/hdf5.hpp>
#endif

#ifdef H5CPP
namespace hdf5
{
  namespace datatype
  {
    template <typename T> class TypeTrait<DiFfRG::complex<T>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = DiFfRG::complex<T>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(2 * sizeof(T));

        type.insert("real", 0, element_type::create(T()));
        type.insert("imag", alignof(T), element_type::create(T()));

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

    template <size_t N, typename T> class TypeTrait<autodiff::Real<N, T>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = autodiff::Real<N, T>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(N * sizeof(autodiff::Real<N, T>));

        type.insert("val", 0, element_type::create(T()));
        for (size_t i = 1; i <= N; ++i) {
          type.insert("d_" + std::to_string(i), i * sizeof(T), element_type::create(T()));
        }

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

    template <typename T, size_t N> class TypeTrait<std::array<T, N>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = std::array<T, N>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(N * sizeof(T));

        for (size_t i = 0; i < N; ++i) {
          type.insert("component " + std::to_string(i), i * sizeof(T), element_type::create(T()));
        }

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

    template <typename T, size_t N>
      requires(!std::is_same_v<std::array<T, N>, DiFfRG::device::array<T, N>>)
    class TypeTrait<DiFfRG::device::array<T, N>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = DiFfRG::device::array<T, N>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(N * sizeof(T));

        for (size_t i = 0; i < N; ++i) {
          type.insert("component " + std::to_string(i), i * sizeof(T), element_type::create(T()));
        }

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };
  } // namespace datatype
} // namespace hdf5
#endif
