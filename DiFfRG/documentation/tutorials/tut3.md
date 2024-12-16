# Tutorial 3: Code generation from Mathematica and Extractors {#tut3}
This tutorial describes, how to generate flow equations from Mathematica and use them in the c-Code.

## Generation of Flow equations
First, one has to import the `DiFfRG` package and set the current directory.
```Mathematica
Get["DiFfRG`"]
SetDirectory[GetDirectory[]];
```
The `GetDirectory[]` command returns the directory of the current notebook of wolfram script.
`DiFfRG` imports, besides its own libraries, the packages `FormTracer` and `QMeSderivation`, which are used for the
diagramatic derivation of the flow equations. The method works as folows:
1. Use `QMeSderivation` to derive a diagramatic description of the 1PI-n-Point Functions
2. Insert the propagators in the flow equations
3. Compute the \f$STr\f$ with `FormTracer`
4. Export the flow equations ()

### 1. QMeSderivation
This will just quickly recap the process, to derive the diagramatic expressions, a more detailed version
can be found [here](https://github.com/QMeS-toolbox/QMeS-Derivation). First, one has to define the
Master-Equation
```Mathematica
fRGEq = {
   "Prefactor" -> {1/2},
   <|"type" -> "Regulatordot", "indices" -> {i, j}|>,
   <|"type" -> "Propagator", "indices" -> {i, j}|>
   };
```
After that, one defines the structure of the model
```Mathematica
fields = <|"bosonic" -> {\[Sigma][p], \[CapitalPi][p, {a}]},

   "fermionic" -> {{qb[p, {d, c, f}], q[p, {d, c, f}]}}|>;

Truncation = {{\[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi]}, {q,
    qb},(* propagators *)
   {qb, q, \[Sigma]}, {qb, q, \[CapitalPi]}, {qb,
    q, \[Sigma], \[Sigma]}, {qb, q, \[CapitalPi], \[CapitalPi]}, {qb,
    q, \[Sigma], \[CapitalPi]}, {qb,
    q, \[Sigma], \[Sigma], \[Sigma]}, {qb,
    q, \[Sigma], \[Sigma], \[CapitalPi]}, {qb,
    q, \[Sigma], \[CapitalPi], \[CapitalPi]}, {qb,
    q, \[CapitalPi], \[CapitalPi], \[CapitalPi]}, (* quark-
   meson scatterings *)
    {\[Sigma], \[Sigma], \[Sigma]}, {\[Sigma], \[CapitalPi], \
\[CapitalPi]}, {\[Sigma], \[Sigma], \[Sigma], \[Sigma]}, {\[Sigma], \
\[Sigma], \[CapitalPi], \[CapitalPi]}, {\[CapitalPi], \[CapitalPi], \
\[CapitalPi], \[CapitalPi]}(* meson scatterings *)
   };

SetupfRG = <|"MasterEquation" -> fRGEq,
   "FieldSpace" -> fields,
   "Truncation" -> Truncation|>;
```
