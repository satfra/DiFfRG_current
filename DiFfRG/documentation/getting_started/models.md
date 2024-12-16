# Numerical Models {#Models}

In DiFfRG the computational setup is fully described by a ***numerical model***. 
A model essentially is a description of a large set of coupled differential equations and some additional methods to handle data output.

In general, we have three components to any flow, which are **FE functions** \f$ u_i(x),\,i\in\{0,\dots,N_f-1\} \f$, **variables** \f$ v_a,\,a\in\{0,\dots,N_v-1\} \f$ and **extractors** \f$ e_b,\,b\in\{0,\dots,N_e-1\} \f$. 

The latter two are just independent variables, whereas the FE functions depend additionaly on a field variable, \f$ u_i(x),\,x\in\mathbb{R}^d \f$. In other words, the FE functions explicitly live on a spatial discretization of the field space.

## Defining a model

Any model you define should at least be inherited from the abstract model class DiFfRG::def::AbstractModel to ensure that all necessary methods are at least defined to do nothing, i.e.
```Cpp
using namespace DiFfRG;
class MyModel : public def::AbstractModel<MyModel>, ...
{
  ...
};
```
Inside the class we can now overwrite all methods from DiFfRG::def::AbstractModel in order to implement the right system of flow equations.
 
## Spatial discretization

The FE functions usually correspond to expansion coefficients in a derivative expansion. As an example consider a bosonic theory as in [this](https://arxiv.org/abs/2305.00816) paper: Treating a purely bosonic theory in first-order derivative expansion, the effective action is given by
\f[\large
  \Gamma_k[\phi] = \int_x \bigg(\frac{1}{2}Z(\rho)(\partial_\mu\phi)^2 + V(\rho) \bigg)\,,
\f]
where \f$ \rho = \phi^2 / 2 \f$.
A flowing reparametrization of the field \f$ \phi \f$ is being performed and is given by
\f[\large
  \dot\phi(x) = \frac{1}{2} \eta(\rho) \phi\,.
\f]
where we introduced the anomalous dimension \f$ \eta = \frac{\partial_{t_+} Z}{Z} \f$.

The flow is then fully parametrized in terms of FE functions
\f{align}{\large
  u_1(x) &= m^2(\rho) = \partial_\rho V(\rho)\,, \\\large
  u_2(x) &= \eta(\rho)\,,
\f}
where we also chose the field \f$ x = \rho \f$. We see here that the FE functions live on a spatial discretization of the d-dimensional field space \f$ \mathbb{R}^d \f$.

With the above ansatz one can quickly compute flow equations from the Wetterich equation,
\f[\large
  k\partial_k \Gamma_k[\Phi] = \frac{1}{2}\text{Tr}\, G_{\alpha\beta}\,k\partial_k R^{\alpha\beta}\,.
\f]
We remark that the time \f[\large t = t_+ = \ln\left(\frac{\Lambda}{k}\right)\,,\f] as used in DiFfRG, is opposite in sign to the RG-time as defined in most literature, \f$t_- = \ln\left(\frac{k}{\Lambda}\right)\f$. This is simply due to many time solvers not accepting negative time arguments.

In order to discretize the flow equations on a finite element space, the flow equations are expressed in the standard differential-algebraic form
\f[\large
  m_i(\partial_t u_j, u_j, x) + \partial_x F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) + s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) = 0\,,
\f]
where \f$ m_i \f$ are called the mass functions, \f$ F_i \f$ the fluxes and \f$ s_i \f$ the sources. The latter two are functions of the FE functions, their derivatives, the field variable, the variables and the extractors.

In principle, the above system of equations can contain both equations containing the time derivatives, i.e. differential components, and equations without time derivatives, i.e. algebraic components. In order to solve the resulting DAEs one is currently restricted to the **SUNDIALS IDA** solver, which is however highly efficient and actually recommended for most cases.

Alternatively, the restricted formulation, allowing only for differential components,
\f[\large
  m_{ij}(x) \partial_t u_j + \partial_x F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) + s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) = 0\,,
\f]
is used for all other provided ODE solvers, i.e. Runge-Kutta methods.

Note, that in the above definitions a change from \f$ t = t_+ \to t_- \f$ simply moves all terms onto the other side, i.e. when calculating the flow equations in the standard \f$t_-\f$, one can still copy and paste everything without changing signs if the mass functions are simply \f$ m_i = \partial_{t_+} u_i = - \partial_{t_-} u_i \f$ (as is default).

The above components of standard form have direct analogues in the abstract ***numerical model*** which must be reimplemented for any system of flow equations. For actual implementation examples, especially regarding the template structure, please take a look at the models contained in the `DiFfRG/models/` folder.

The relevant methods are also documented in DiFfRG::def::AbstractModel and read as follows:

- The mass function \f$m_i(\partial_t u_j, u_j, x)\f$ is implemented in the method 
```Cpp
  template <int dim, typename NumberType, typename Vector, typename Vector_dot, size_t n_fe_functions>
  void mass(std::array<NumberType, n_fe_functions> &m_i, const Point<dim> &x, const Vector &u, const Vector_dot &u_dot) const;
```  
Note, that the precise template structure is not important, the only important thing is that the types are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector, and Vector_dot template parameters, as these can differ between calls.  
The `m_i` argument is the resulting mass function \f$m_i\f$, with \f$N_f\f$ components. This method should fill the `m_i` argument with the desired structure of the flow equation. `x` is a d-dimensional array of field coordinates, and both `u` (~\f$u_i(x)\f$) and `u_dot` (~\f$\partial_t u_i(x)\f$) have \f$N_f\f$ components.  
The standard implementation of this method simply sets \f$m_i = \partial_t u_i\f$.  
 
.
- If not using a DAE, the mass matrix \f$m_{ij}(x)\f$ is implemented in the method 
```Cpp
  template <int dim, typename NumberType, size_t n_fe_functions>
  void mass(std::array<std::array<NumberType, n_fe_functions>, M::Components::count_fe_functions()> &m_ij, const Point<dim> &x) const;
```
The `m_ij` argument is the resulting mass matrix \f$m_{ij}\f$, with \f$N_f\f$ components in each dimension. This method should fill the `mass` argument with the desired structure of the flow equation. `x` is a d-dimensional array of field coordinates.
The standard implementation of this method simply sets \f$m_{ij} = \delta_{ij}\f$.

.
- The flux function \f$F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)\f$ is given by
```Cpp
  template <int dim, typename NumberType, typename Solution, size_t n_fe_functions>
  void flux(std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i, const Point<dim> &x, const Solution &sol) const;
```
Onve again, it is necessary to leave the `NumberType` and `Solution` templates, whereas the rest can be dropped.
`F_i` has \f$N_f\f$ components, `x` gives the coordinate in field space and `sol` contains all other arguments of the flux function. In practice, `sol` is a `std::tuple<...>` which contains
  0. the array u_j
  1. the array of arrays \f$\partial_x u_j\f$
  2. the array of arrays of arrays \f$\partial_x^2 u_j\f$
  3. the array of extractors \f$e_b\f$
Lastly, the variables are communicated separately to the model, see the ***Other variables***-section below
The standard implementation of this method simply sets \f$F_i = 0\f$.

.
- The source function \f$s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)\f$ is given by
```Cpp
  template <int dim, typename NumberType, typename Solution, size_t n_fe_functions>
  void source(std::array<NumberType, n_fe_functions> &s_i, const Point<dim> &x, const Solution &sol) const;
```
Again, it is necessary to leave the `NumberType` and `Solution` templates, whereas the rest can be dropped.
`s_i` has \f$N_f\f$ components, `x` gives the coordinate in field space and `sol` contains all other arguments of the flux function, with the layout as explained above in the flux case.
The standard implementation of this method simply sets \f$s_i = 0\f$.

Picking up the example from above, we can now sketch the implementation of the numerical model as follows:
```Cpp
using namespace DiFfRG;

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

class MyModel : public def::AbstractModel<MyModel, Components>,
                public def::fRG,                    // this handles the fRG time
                ...
{
public:
  MyModel(const JSONValue& json) : def::fRG(json), prm(json) {}

  template <int dim, typename NumberType, typename Vector, typename Vector_dot>
  void mass(std::array<NumberType, Components::count_fe_functions()> &m_i, const Point<dim> &x, const Vector &u, const Vector_dot &u_dot) const
  {
    m_i[idxf("u")] = u_dot[idxf("u")];
    m_i[idxf("v")] = -u[idxf("v")];
  }

  template <int dim, typename NumberType, typename Solution>
  void flux(std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> &F_i, const Point<dim> &x, const Solution &sol) const;
  {
    F_i[idxf("u")][0] = ...; // Flux of m^2
    F_i[idxf("v")][0] = ...; // Flux of eta
  }

  template <int dim, typename NumberType, typename Solution, typename M = Model>
  void source(std::array<NumberType, M::Components::count_fe_functions(0)> &s_i, const Point<dim> &x, const Solution &sol) const
  {
    s_i[idxf("u")] = ...; // Source of m^2
    s_i[idxf("v")] = ...; // Source of eta
  }
};
```

### Assemblers and discretizations

The actual numerical calculation of the flow equations (rather, their weak form) is done by the so-called assemblers. These are responsible for the actual discretization of the flow equations on the finite element space. In DiFfRG, we provide a set of assemblers for different discretizations, which are all derived from the abstract assembler class DiFfRG::AbstractAssembler.

Although all require at least the above interface methods, certain additional methods are required for certain discretizations. For example, the discontinuous Galerkin (DG) assemblers require the implementation of the numerical fluxes, and both discontinuous and continuous Galerkin (CG, also called simply FEM here) assemblers require the implementation of the boundary condition fluxes.

To understand the underlying numerics, see e.g. this [review](https://www3.nd.edu/~zxu2/acms60790S15/DG-general-approach.pdf) and also the excellent [deal.ii tutorials](https://www.dealii.org/developer/doxygen/deal.II/Tutorial.html).

For further reference, please refer to the documentation of the respective assemblers.
- DiFfRG::DG::Assembler
- DiFfRG::dDG::Assembler
- DiFfRG::CG::Assembler

Underlying the assemblers are the actual discretizations, which are implemented in the DiFfRG::discretization namespace. These are responsible for the actual discretization of the field space, i.e. the construction of the finite element space. In DiFfRG, we provide a set of discretizations for different finite element spaces.
- DiFfRG::DG::Discretization
- DiFfRG::CG::Discretization

### Running

Putting everything together, we can write a straightforward main function to run the flow equations:

```Cpp

using namespace dealii;
using namespace DiFfRG;

int main(int argc, char *argv[])
{
  // declare/get all needed parameters and parse from the CLI
  ConfigurationHelper config_helper(argc, argv);
  // declare all parameters that are going to be read from the config file
  declare_discretization_parameters(config_helper.get_parameter_handler());
  declare_timestepping_parameters(config_helper.get_parameter_handler());
  declare_physics_parameters(config_helper.get_parameter_handler());
  // parse the config file and the CLI parameters
  config_helper.parse();
  // read the parameters into data structures
  DiscretizationParameters d_prm = get_discretization_parameters(config_helper.get_parameter_handler());
  TimesteppingParameters t_prm = get_timestepping_parameters(config_helper.get_parameter_handler());
  PhysicalParameters p_prm = get_physical_parameters(config_helper.get_parameter_handler());

  // Make choices for types: The model, its discretization, the assembler and the timestepper
  using Model = MyModel;
  using NumberType = double;
  constexpr uint dim = 1;
  using Discretization = DG::Discretization<Model::Components, NumberType, Mesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, dim>;

  // Define the objects needed to run the simulation
  Model model(p_prm);
  Mesh<dim> mesh(d_prm);
  Discretization discretization(mesh, d_prm);
  Assembler assembler(discretization, model, d_prm.overintegration, d_prm.threads, d_prm.batch_size);
  DataOutput<dim> data_out(discretization.get_dof_handler(), config_helper.get_top_folder(), config_helper.get_output_name(), config_helper.get_output_folder(),
                           d_prm.output_subdivisions);
  MeshAdaptor mesh_adaptor(assembler, d_prm);
  TimeStepper time_stepper(&assembler, data_out, &mesh_adaptor, t_prm);

  // Set up the initial condition
  FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  // Start the timestepping
  Timer timer;
  time_stepper.run(initial_condition.spatial_data(), 0., t_prm.final_time);

  // Print a bit of exit information to the logger.
  deallog << "Program finished." << std::endl;
  return 0;
}
```

Here we have chosen a `dDG` discretization, which is a discontinuous Galerkin discretization with a direct discontinuous Galerkin assembler. The timestepper is the SUNDIALS IDA solver, which is the recommended solver for most cases.
For the `dDG` discretization it is also necessary to supply a numerical flux, which can be done by modifying the numerical model as follows:
```Cpp
class MyModel : public def::AbstractModel<MyModel>,
                public def::fRG,                    // this handles the fRG time
                public def::LLFFlux<largeN>,        // use a LLF numflux
                public def::FlowBoundaries<largeN>, // use Inflow/Outflow boundaries
                public def::AD<largeN>              // define all jacobians per AD
{
  ...
};
```
Here, the local Lax-Friedrichs flux has been used for the numerical fluxes and the boundaries have been defined to be inflow-/outflow. We have also chosen to use the autodiff functionality for the calculation of the jacobians.

## Other variables