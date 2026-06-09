(* ::Package:: *)

(* DiFfRG / FunKit code generation for the fully momentum-dependent (angle-resolved)
   Yang-Mills truncation. Ported to the current FunKit API, referencing
   Examples/YangMills/SP/Yang-Mills.m (propagators + codegen skeleton) and
   qcd-codes/sm/Quark-Meson.m (angle-resolved 3D-grid vertex kernels:
   P3Param, DeclareAnglesP34Dpqr, LogLogLinCoordinates, {p1,p2,cosp1p2}).
   Run headless with:  wolfram -script Yang-Mills.m  *)


(* ::Input::Initialization:: *)
Get["DiFfRG`"]
SetDirectory[GetDirectory[]];
DefineFormExecutable["/usr/bin/tform -w16"]


fields= <|
"Commuting"-> {A[p,{v, c}]},
"Grassmann"->{{cb[p,{c}],c[p,{c}]}}
|>;


truncation=<|
GammaN->{{A,A},{A,A,A},{A,A,A,A},{A,cb,c},{cb,c}},
Propagator->{{A,A},{cb,c}},Rdot->{{A,A},{cb,c}},
S->{{A,A},{A,A,A},{A,A,A,A},{cb,c},{cb,c,A}},
Field->{{}}
|>;


bases=<|
GammaN->{{A,A}->{"AA",1},{A,A,A}->"AAAClass",{A,A,A,A}->"AAAAClass",{A,cb,c}->{"Acbc",1},{cb,c}->"cbc"},
S->{{A,A}->{"AA",1},{A,A,A}->"AAAClass",{A,A,A,A}->"AAAAClass",{A,cb,c}->{"Acbc",1},{cb,c}->"cbc"},
Propagator->{{A,A}->{"AA",1},{cb,c}->"cbc"},
Rdot->{{A,A}->{"AA",1},{cb,c}->"cbc"}
|>;


diagramStyling=<|"Styles"->{A->{Orange},c->{Black,Dashed}}|>;
FSetTexStyles[cb->"\\bar{c}"];


Setup=<|
"FieldSpace"->fields,
"Truncation"->truncation,
"FeynmanRules"->bases,
"DiagramStyling"->diagramStyling
|>;
FSetGlobalSetup[Setup];


SP3Patt[p1e_,p2e_,p3e_]:={Sqrt[(sp[p1,p1]+sp[p2,p2]+sp[p3,p3])/3]}/.{p1:>p1e,p2:>p2e,p3:>p3e}//UseLorentzLinearity//FullSimplify;
SP4Patt[p1e_,p2e_,p3e_,p4e_]:={Sqrt[(sp[p1,p1]+sp[p2,p2]+sp[p3,p3]+sp[p4,p4])/4]}/.{p1:>p1e,p2:>p2e,p3:>p3e,p4:>p4e}//UseLorentzLinearity//FullSimplify;

(* Symmetric-triangle (S0,S1,SPhi) parametrisation of the three-point function, exactly as in the
   legacy angle-resolved truncation (TB3PToS0S1SPhi). S0 is the overall scale (log-gridded), S1 the
   shape (linear, 0 at the symmetric point, ->1 at collinear) and SPhi an angle (linear). The
   transverse projector's collinear 1/(cos^2-1) then sits only at the single grid CORNER
   (S1->1 AND SPhi->0), not on a whole face as with raw (|p1|,|p2|,cos) -- which is what makes the
   angle-resolved flow gridded this way numerically stable (the legacy code is tunable like this). *)
transf3PTo=TB3PToS0S1SPhi[p1,p2,p3,l1,S0,S1,SPhi];
transf3PFrom=TB3PFromS0S1SPhi[p1,p2,p3,S0,S1,SPhi];
P3Patt[p1e_,p2e_,p3e_]:={transf3PFrom[S0],transf3PFrom[S1],transf3PFrom[SPhi]}/.{p1:>p1e,p2:>p2e,p3:>p3e}//UseLorentzLinearity//FullSimplify;

(* Tadpole (Qk) parametrisation: the 4-gluon in the gluon self-energy tadpole depends on one
   external (p) and one loop (q) momentum; tadpolePatt maps that (external,loop) pair onto the same
   (S0,S1,SPhi) grid. This is the legacy treatment -- the gluon self-energy must be fed the
   angle-resolved ZA4tadpole(external,loop), NOT the symmetric-point ZA4SP. *)
transfTadpoleTo=TB3PToS0S1SPhiQk[p1,p2,q,S0,S1,SPhi];(* the Qk transform only parametrises the loop when it is named q *)
transfTadpoleFrom=TB3PFromS0S1SPhiQk[p1,p2,S0,S1,SPhi];
tadpolePatt[p1e_,p2e_]:={transfTadpoleFrom[S0],transfTadpoleFrom[S1],transfTadpoleFrom[SPhi]}/.{p1:>p1e,p2:>p2e}//UseLorentzLinearity//FullSimplify;


(* Rules common to all flows (propagators, ghost-gluon and 3-gluon vertices, regulators). The
   4-gluon rule differs by context and is appended separately below. *)
commonDressing={
dressing[GammaN,{cb,c},1,{p1_,p2_}]:>-Zc[Sqrt[sp[p2,p2]]]sp[p2,p2],
dressing[GammaN,{A,A},1,{p1_,p2_}]:>ZA[Sqrt[sp[p2,p2]]]sp[p2,p2],

dressing[InverseProp,{cb,c},1,{p1_,p2_}]:>-(Zc[Sqrt[sp[p2,p2]]]sp[p2,p2]+RB[k^2,sp[p2,p2]]Zc[k]),
dressing[InverseProp,{A,A},1,{p1_,p2_}]:>ZA[Sqrt[sp[p2,p2]]]sp[p2,p2]+RB[k^2,sp[p2,p2]]ZA[evP],

(* Angle-resolved 3-point vertices on the (S0,S1,SPhi) grid; the feed carries the loop momentum. *)
dressing[GammaN,{A,cb,c},1,{p1_,p2_,p3_}]:>ZAcbc[p1,p2],
dressing[GammaN,{A,A,A},1,{p1_,p2_,p3_}]:>ZA3[p1,p2],

ZAcbc[p1_,p2_]:>ZAcbc@@P3Patt[p1,p2,-p1-p2],(* angle-resolved ghost-gluon vertex (3D) *)
ZA3[p1_,p2_]:>ZA3@@P3Patt[p1,p2,-p1-p2],(* angle-resolved 3-gluon vertex (3D) *)
ZA4SP[p1_,p2_,p3_]:>ZA4SP@@SP4Patt[p1,p2,p3,-p1-p2-p3],

nZA->6,
evP:>(k^nZA+1)^(1/nZA),
devP:>k^(-1+nZA) (1+k^nZA)^(-1+1/nZA),
dressing[Rdot,{A,A},1,{p1_,p2_}]:>ZA[evP]RBdot[k^2,sp[p2,p2]]+RB[k^2,sp[p2,p2]](dtZA[evP]+k*devP*(ZA[1.02evP]-ZA[evP])/(0.02*evP)),
dressing[Rdot,{cb,c},1,{p1_,p2_}]:>Zc[k]RBdot[k^2,sp[p2,p2]]+RB[k^2,sp[p2,p2]](dtZc[k]+k (Zc[1.02*k]-Zc[k])/(0.02*k))
};

(* Standard rules (used for the vertex flows): 4-gluon -> symmetric-point ZA4SP. *)
dressingRules=ReplaceRepeated[#,Join[{dressing[GammaN,{A,A,A,A},1,{p1_,p2_,p3_,p4_}]:>ZA4SP[p1,p2,p3]},commonDressing]]&;

(* Gluon self-energy rules: the 4-gluon tadpole is fed via the angle-resolved ZA4tadpole, taking one
   external (leg 1) and one loop (leg 3) momentum, exactly as the legacy PreTraceRulesGluonsTadpole. *)
dressingRulesAA=ReplaceRepeated[#,Join[{dressing[GammaN,{A,A,A,A},1,{p1_,p2_,p3_,p4_}]:>ZA4tadpole@@tadpolePatt[p1,p3]},commonDressing]]&;

FSetSymmetricDressing[GammaN,{A,A}]


(* Parametrisation of scalar products for the 1D (propagator) kernels. *)
PropParam[expr_]:=UseLorentzLinearity[expr]//.{
lf1->l1,
sp[p1,p1]->p^2,sp[l1,l1]->l1^2,
sp[l1,p1]->l1 p cos[p,l1],
sp[p1,l1]->l1 p cos[p,l1],
Sqrt[a_^2]:>a,(a_^2)^(n_/2):>a^n,
cos[l1,p]:>cos1
};

(* Symmetric-point parametrisation for ZA4SP (collapses external momenta to one scale). *)
SP3FormRule=FMakeSPFormRule[{l1,lf1},p,{p1,p2,p3}];
SP4FormRule=FMakeSPFormRule[{l1,lf1},p,{p1,p2,p3,p4}];
SPParam[expr_]:=UseLorentzLinearity[expr]//.{
lf1->l1,
sp[p,p]->p^2,sp[l1,l1]->l1^2,
sp[l1,p1]->p l1 cos[l1,p1],
sp[l1,p2]->p l1 cos[l1,p2],
sp[l1,p3]->p l1 cos[l1,p3],
sp[l1,p4]->p l1 cos[l1,p4],
Sqrt[a_^2]:>a,(a_^2)^(n_/2):>a^n,(a_^2)^(n_/2):>a^n,Power[Power[l1_,2],Rational[n_,2]]:>l1^n,
cos[l1,p1]:>cosl1p1,
cos[l1,p2]:>cosl1p2,
cos[l1,p3]:>cosl1p3,
cos[l1,p4]:>cosl1p4
};

(* Angle-resolved parametrisation for the 3D vertices (external grid p1,p2,cosp1p2).
   Full 4D products throughout (vacuum); the loop-external cosines cosl1p1/cosl1p2 are
   declared in the KernelBody via DeclareAnglesP34Dpqr. *)
P3Param[expr_]:=UseLorentzLinearity[expr]//.{
lf1->l1,
sp[p1,p1]->p1^2,
sp[p2,p2]->p2^2,
sp[p1,p2]->p1 p2 cosp1p2,
sp[p2,p1]->p1 p2 cosp1p2,
sp[l1,l1]->l1^2,
sp[l1,p1]->l1 p1 cosl1p1,
sp[p1,l1]->l1 p1 cosl1p1,
sp[l1,p2]->l1 p2 cosl1p2,
sp[p2,l1]->l1 p2 cosl1p2,
Sqrt[a_^2]:>a,(a_^2)^(n_/2):>a^n,Power[Power[l1_,2],Rational[n_,2]]:>l1^n,
cos[l1,p1]:>cosl1p1,
cos[l1,p2]:>cosl1p2
};

SetNc[3]
$Assumptions=k>0&&p>0&&p1>0&&p2>0&&l1>0&&q>0&&-1<cos1<1&&-1<cos2<1&&-1<cos3<1&&-1<cosp1p2<1;


(* ::Input::Initialization:: *)
interpolatorType="SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory>";(* Spline (C^2) like the working SP example -- linear interpolation of the propagators injects kinks that block the deep-IR scaling trajectory *)
interpolator3DType="LinearInterpolatorND<double, LogLinLinCoordinates, GPU_memory>";

kernelParameterList={
<|"Name"->"k","Type"->"double"|>,
(*strong couplings*)
<|"Name"->"ZA3","Type"->interpolator3DType,"Const"->True,"Reference"->True|>,(* angle-resolved 3D vertex *)
<|"Name"->"ZAcbc","Type"->interpolator3DType,"Const"->True,"Reference"->True|>,(* angle-resolved 3D vertex *)
<|"Name"->"ZA4SP","Type"->interpolatorType,"Const"->True,"Reference"->True|>,
<|"Name"->"ZA4tadpole","Type"->interpolator3DType,"Const"->True,"Reference"->True|>,(* angle-resolved tadpole *)
(*ghost propagator*)
<|"Name"->"dtZc","Type"->interpolatorType,"Const"->True,"Reference"->True|>,
<|"Name"->"Zc","Type"->interpolatorType,"Const"->True,"Reference"->True|>,
(*glue propagator*)
<|"Name"->"dtZA","Type"->interpolatorType,"Const"->True,"Reference"->True|>,
<|"Name"->"ZA","Type"->interpolatorType,"Const"->True,"Reference"->True|>
};

SP4Defs=DeclareSymmetricPoints4DP4[l1,p,{p1,p2,p3,p4}];
SP3Defs=DeclareSymmetricPoints4DP3[l1,p,{p1,p2,p3}];
(* Angle-resolved loop-external cosines for the 3D vertices. *)
P3AngDef=DeclareAnglesP34Dpqr[l1,p1,p2];

(* KernelBody for the (S0,S1,SPhi)-gridded 3-point vertices: declare the external magnitudes
   p1,p2, the external angle cosp1p2 and the loop-external cosines cosl1p1,cosl1p2 in terms of the
   grid coordinates (S0,S1,SPhi) and the loop integration angles (cos1,cos2). Mirrors the legacy
   DeclareAngles[q,p1,p2]; phi drops out so a 2-angle integrator suffices. *)
S0S1SPhiDef=Module[{rp1,rp2,rcosp1p2,rcosl1p1,rcosl1p2},
  rcosl1p1=transf3PTo[sp[p1,l1]/(l1 Sqrt[sp[p1,p1]])]//FullSimplify;
  rcosl1p2=transf3PTo[sp[p2,l1]/(l1 Sqrt[sp[p2,p2]])]//FullSimplify;
  rcosp1p2=transf3PTo[sp[p1,p2]/Sqrt[sp[p1,p1] sp[p2,p2]]]//FullSimplify;
  rp1=transf3PTo[Sqrt[sp[p1,p1]]]//FullSimplify;
  rp2=transf3PTo[Sqrt[sp[p2,p2]]]//FullSimplify;
  "const double p1 = "<>FunKit`CppForm[rp1]<>";\n"<>
  "const double p2 = "<>FunKit`CppForm[rp2]<>";\n"<>
  "const double cosp1p2 = "<>FunKit`CppForm[rcosp1p2]<>";\n"<>
  "const double cosl1p1 = "<>FunKit`CppForm[rcosl1p1]<>";\n"<>
  "const double cosl1p2 = "<>FunKit`CppForm[rcosl1p2]<>";"];

(* KernelBody for the angle-resolved 4-gluon tadpole flow: same as S0S1SPhiDef but with the Qk
   (one external + one loop momentum) parametrisation. Mirrors the legacy DeclareAnglesTadpole. *)
TadpoleDef=Module[{rp1,rp2,rcosp1p2,rcosl1p1,rcosl1p2},
  rcosl1p1=transfTadpoleTo[sp[p1,q]/(q Sqrt[sp[p1,p1]])]//FullSimplify;(* = cos(p1,loop); loop-magnitude-free *)
  rcosl1p2=transfTadpoleTo[sp[p2,q]/(q Sqrt[sp[p2,p2]])]//FullSimplify;
  rcosp1p2=transfTadpoleTo[sp[p1,p2]/Sqrt[sp[p1,p1] sp[p2,p2]]]//FullSimplify;
  rp1=transfTadpoleTo[Sqrt[sp[p1,p1]]]//FullSimplify;
  rp2=transfTadpoleTo[Sqrt[sp[p2,p2]]]//FullSimplify;
  "const double p1 = "<>FunKit`CppForm[rp1]<>";\n"<>
  "const double p2 = "<>FunKit`CppForm[rp2]<>";\n"<>
  "const double cosp1p2 = "<>FunKit`CppForm[rcosp1p2]<>";\n"<>
  "const double cosl1p1 = "<>FunKit`CppForm[rcosl1p1]<>";\n"<>
  "const double cosl1p2 = "<>FunKit`CppForm[rcosl1p2]<>";"];


FSetCacheDirectory[GetDirectory[]<>"/TraceCache/"]


FSetRegisterSize[64]


(* ============================ Gluon propagator ZA ============================ *)
fRGAA=FTakeDerivatives[WetterichEquation,{A[i1],A[i2]}]//FTruncate//FPlot//FRoute//FPrint;

traceExprAA=FTerm[TBGetProjector["AA",1,{i1,i2}/.fRGAA["1-Loop"]["ExternalIndices"]]]**(fRGAA["1-Loop"]["Expression"]/.FMakeDiagrammaticRules[]);
(* dressingRulesAA feeds the 4-gluon tadpole via the angle-resolved ZA4tadpole(external,loop). *)
FlowAA=FormTrace[traceExprAA]//dressingRulesAA//PropParam//Simplify;

MakeKernel[FlowAA/p^2,
"Name"->"ZA",
"Integrator"->"Integrator_p2_1ang",
"d"->4,
"AD"->False,
"ctype"->"double",
"Device"->"GPU",
"Type"->"double",
"Parameters"->kernelParameterList,
"IntegrationVariables"->{"l1","cos1"},
"Coordinates"->{"LogarithmicCoordinates1D<double>"},
"CoordinateArguments"->{"p"}]
UpdateFlows["YangMillsFlows"]


(* ============================ Ghost propagator Zc ============================ *)
fRGcbc=FTakeDerivatives[WetterichEquation,{cb[i1],c[i2]}]//FTruncate//FPlot//FRoute//FPrint;

traceExprcbc=FTerm[TBGetProjector["cbc",1,{i1,i2}/.fRGcbc["1-Loop"]["ExternalIndices"]]]**(fRGcbc["1-Loop"]["Expression"]/.FMakeDiagrammaticRules[]);
Flowcbc=FormTrace[traceExprcbc]//dressingRules//PropParam//Simplify;

MakeKernel[-(Flowcbc/p^2),
"Name"->"Zc",
"Integrator"->"Integrator_p2_1ang",
"d"->4,
"AD"->False,
"ctype"->"double",
"Device"->"GPU",
"Type"->"double",
"Parameters"->kernelParameterList,
"IntegrationVariables"->{"l1","cos1"},
"Coordinates"->{"LogarithmicCoordinates1D<double>"},
"CoordinateArguments"->{"p"}]
UpdateFlows["YangMillsFlows"]


(* ===================== Ghost-gluon vertex ZAcbc (angle-resolved, 3D) ===================== *)
fRGAcbc=FTakeDerivatives[WetterichEquation,{A[i1],cb[i2],c[i3]}]//FTruncate//FSimplify//FPlot//FRoute//FPrint;

projectorAcbc=FTerm[TBGetProjector["Acbc",1,{i1,i2,i3}/.fRGAcbc["1-Loop"]["ExternalIndices"]]];
traceExprAcbc=projectorAcbc**(fRGAcbc["1-Loop"]["Expression"]/.FMakeDiagrammaticRules[]);

FlowAcbc=FormTrace["ZAcbc",traceExprAcbc]//dressingRules//P3Param//Simplify;

MakeKernel[FlowAcbc,
"Name"->"ZAcbc",
"Integrator"->"Integrator_p2_4D_2ang",
"d"->4,
"AD"->False,
"ctype"->"double",
"Device"->"GPU",
"Type"->"double",
"Parameters"->kernelParameterList,
"KernelBody"->S0S1SPhiDef,
"IntegrationVariables"->{"l1","cos1","cos2"},
"Coordinates"->{"LogLinLinCoordinates"},
"CoordinateArguments"->{"S0","S1","SPhi"}]
UpdateFlows["YangMillsFlows"]


(* ===================== Three-gluon vertex ZA3 (angle-resolved, 3D) ===================== *)
fRGA3=FTakeDerivatives[WetterichEquation,{A[i1],A[i2],A[i3]}]//FTruncate//FPlot//FRoute//FPrint;

projectorA3=FTerm[TBGetProjector["AAAClassTrans",1,{i1,i2,i3}/.fRGA3["1-Loop"]["ExternalIndices"]]];
traceExprA3=projectorA3**(fRGA3["1-Loop"]["Expression"]/.FMakeDiagrammaticRules[]);
FlowA3=FormTrace["ZA3",traceExprA3]//dressingRules//P3Param//Simplify;

MakeKernel[FlowA3,
"Name"->"ZA3",
"Integrator"->"Integrator_p2_4D_2ang",
"d"->4,
"AD"->False,
"ctype"->"double",
"Device"->"GPU",
"Type"->"double",
"Parameters"->kernelParameterList,
"KernelBody"->S0S1SPhiDef,
"IntegrationVariables"->{"l1","cos1","cos2"},
"Coordinates"->{"LogLinLinCoordinates"},
"CoordinateArguments"->{"S0","S1","SPhi"}]
UpdateFlows["YangMillsFlows"]


(* ===================== Four-gluon tadpole ZA4tadpole (angle-resolved, Qk) ===================== *)
(* Same 4-gluon diagrams, but at the tadpole leg configuration {p1,p2,-p1,-p2} and gridded over the
   (S0,S1,SPhi) of the (external,loop) pair. This is the dressing fed into the gluon self-energy. *)
fRGA4t=FTakeDerivatives[WetterichEquation,{A[i1],A[i2],A[i3],A[i4]}]//FTruncate//FPlot//FRoute//FPrint;

projectorA4t=(FTerm[TBGetProjector["AAAAClassTrans",1,{i1,i2,i3,i4}/.fRGA4t["1-Loop"]["ExternalIndices"]]])//.{p3->-p1,p4->-p2}//UseLorentzLinearity//FullSimplify;
traceExprA4t=(projectorA4t**(fRGA4t["1-Loop"]["Expression"]/.FMakeDiagrammaticRules[]))//.{p3->-p1,p4->-p2}//UseLorentzLinearity//FullSimplify;
FlowA4t=FormTrace["ZA4tadpole",traceExprA4t]//dressingRules//P3Param//Simplify;

MakeKernel[FlowA4t,
"Name"->"ZA4tadpole",
"Integrator"->"Integrator_p2_4D_2ang",
"d"->4,
"AD"->False,
"ctype"->"double",
"Device"->"GPU",
"Type"->"double",
"Parameters"->kernelParameterList,
"KernelBody"->TadpoleDef,
"IntegrationVariables"->{"l1","cos1","cos2"},
"Coordinates"->{"LogLinLinCoordinates"},
"CoordinateArguments"->{"S0","S1","SPhi"}]
UpdateFlows["YangMillsFlows"]


(* ===================== Four-gluon symmetric point ZA4SP ===================== *)
fRGA4=FTakeDerivatives[WetterichEquation,{A[i1],A[i2],A[i3],A[i4]}]//FTruncate//FPlot//FRoute//FPrint;

projectorA4=FTerm[TBGetProjector["AAAAClassTrans",1,{i1,i2,i3,i4}/.fRGA4["1-Loop"]["ExternalIndices"]]]//TBProjectToSymmetricPoint[#,l1,p,p1,p2,p3,p4]&//Simplify;
traceExprA4=projectorA4**(fRGA4["1-Loop"]["Expression"]/.FMakeDiagrammaticRules[])//TBProjectToSymmetricPoint[#,l1,p,p1,p2,p3,p4]&;
FlowA4=FormTrace["ZA4SP",traceExprA4,{},SP4FormRule]//dressingRules//TBProjectToSymmetricPoint[#,l1,p,p1,p2,p3,p4]&//SPParam//Simplify;

MakeKernel[FlowA4,
"Name"->"ZA4SP",
"Integrator"->"Integrator_p2_4D_3ang",
"d"->4,
"AD"->False,
"ctype"->"double",
"Device"->"GPU",
"Type"->"double",
"Parameters"->kernelParameterList,
"KernelBody"->SP4Defs,
"IntegrationVariables"->{"l1","cos1","cos2","phi"},
"Coordinates"->{"LogarithmicCoordinates1D<double>"},
"CoordinateArguments"->{"p"}]
UpdateFlows["YangMillsFlows"]
