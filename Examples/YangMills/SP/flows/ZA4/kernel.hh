#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA4_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const auto &l1, const auto &cos1, const auto &cos2, const auto &phi, const auto &p, const double &k,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZAcbc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const double cosl1p1 = cos1;
      const double cosl1p2 = ((-1.) * (cos1) + (sqrt(3. + (-3.) * (powr<2>(cos1)))) * (cos2)) * (0.5);
      const double cosl1p3 = ((-1.) * (cos1) + (-1.) * ((sqrt(3. + (-3.) * (powr<2>(cos1)))) * (cos2))) * (0.5);
      const auto _repl1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl2 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl5 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _repl6 = ZA(l1);
      const auto _repl7 =
          RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p));
      const auto _repl8 = ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p)));
      const auto _repl9 =
          RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (1.333333333333333) * (powr<2>(p)));
      const auto _repl10 =
          ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (1.333333333333333) * (powr<2>(p))));
      const auto _repl11 = ZA3((0.816496580927726) *
                               (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl12 =
          ZA3((0.4714045207910317) *
              (sqrt((3.) * (powr<2>(l1)) + (-3.) * ((cosl1p1 + cosl1p2) * (2.) + cosl1p3) * ((l1) * (p)) +
                    (5.) * (powr<2>(p)))));
      const auto _repl13 = Zc(k);
      return (0.163265306122449) *
                 ((-50.) * (_repl4 + (-1.) * (_repl5)) * ((_repl2) * (powr<6>(k))) +
                  (_repl1) * (1. + powr<6>(k)) * (_repl2) + (_repl3) * (1. + powr<6>(k)) * (_repl4)) *
                 ((_repl11) *
                  ((-243.) *
                       ((3.) * (powr<4>(cosl1p1)) + (-6.) * (powr<4>(cosl1p2)) +
                        (-12.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                        (3.) * (cosl1p2 + (3.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                        (2. + (-3.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                        (-1. + (3.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p3)) +
                        (cosl1p2) * (2. + (3.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                        (-1. + (-3.) * (powr<2>(cosl1p2)) + (5.) * ((cosl1p2) * (cosl1p3)) +
                         (6.) * (powr<2>(cosl1p3))) *
                            (powr<2>(cosl1p1)) +
                        ((2.) * (cosl1p2) + (-12.) * (powr<3>(cosl1p2)) + (-4.) * (cosl1p3) +
                         (-10.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (5.) * ((cosl1p2) * (powr<2>(cosl1p3))) +
                         (9.) * (powr<3>(cosl1p3))) *
                            (cosl1p1)) *
                       (powr<10>(l1)) +
                   (81.) *
                       ((54.) * (powr<5>(cosl1p1)) + (-72.) * (powr<5>(cosl1p2)) +
                        (-180.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
                        (90.) * (cosl1p2 + (2.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
                        (6. + (-108.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
                        (-6.) *
                            (1. + (3.) * (powr<2>(cosl1p2)) + (-36.) * ((cosl1p2) * (cosl1p3)) +
                             (-27.) * (powr<2>(cosl1p3))) *
                            (powr<3>(cosl1p1)) +
                        (3.) * (5. + (6.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                        ((41.) * (cosl1p2) + (-252.) * (powr<3>(cosl1p2)) + (-59.) * (cosl1p3) +
                         (-138.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (192.) * ((cosl1p2) * (powr<2>(cosl1p3))) +
                         (198.) * (powr<3>(cosl1p3))) *
                            (powr<2>(cosl1p1)) +
                        (2.) * (-2. + (9.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                        (4. + (-3.) * (powr<2>(cosl1p3)) + (54.) * (powr<4>(cosl1p3))) * (cosl1p2) +
                        ((-252.) * (powr<4>(cosl1p2)) + (-408.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                         (192.) * ((cosl1p2) * (powr<3>(cosl1p3))) +
                         (53. + (-54.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         (-47. + (108.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p3))) *
                            (cosl1p1)) *
                       ((powr<9>(l1)) * (p)) +
                   (27.) *
                       (16. + (-39.) * (powr<2>(cosl1p1)) + (-207.) * (powr<4>(cosl1p1)) +
                        (-333.) * (powr<6>(cosl1p1)) +
                        (-1.) *
                            ((1224.) * (powr<5>(cosl1p1)) + (1350.) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                             (9.) * (15. + (164.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p1)) +
                             (6.) * (-53. + (171.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p1)) * (cosl1p3)) +
                             (-28. + (135.) * (powr<2>(cosl1p3)) + (9.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                             (3.) * (-18. + (77.) * (powr<2>(cosl1p3)) + (84.) * (powr<4>(cosl1p3))) * (cosl1p1)) *
                            (cosl1p3) +
                        ((-3.) * (44. + (231.) * (powr<2>(cosl1p1)) + (258.) * (powr<4>(cosl1p1))) * (cosl1p1) +
                         (-1.) *
                             (-10. + (2220.) * (powr<4>(cosl1p1)) + (2172.) * ((powr<3>(cosl1p1)) * (cosl1p3)) +
                              (189.) * (powr<2>(cosl1p3)) + (126.) * (powr<4>(cosl1p3)) +
                              (3.) * (-23. + (344.) * (powr<2>(cosl1p3))) * ((cosl1p1) * (cosl1p3)) +
                              (3.) * (89. + (760.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p1))) *
                             (cosl1p3) +
                         (-65. + (-225.) * (powr<4>(cosl1p1)) + (234.) * (powr<4>(cosl1p2)) +
                          (702.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (153.) * (powr<2>(cosl1p3)) +
                          (-225.) * (powr<4>(cosl1p3)) +
                          (12.) * ((138.) * (cosl1p2) + (11.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                          (3.) *
                              (-173. + (849.) * (powr<2>(cosl1p2)) + (1208.) * ((cosl1p2) * (cosl1p3)) +
                               (-24.) * (powr<2>(cosl1p3))) *
                              (powr<2>(cosl1p1)) +
                          (36.) * (17. + powr<2>(cosl1p3)) * ((cosl1p2) * (cosl1p3)) +
                          (9.) * (38. + (67.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                          (6.) *
                              ((225.) * (powr<3>(cosl1p2)) + (73.) * (cosl1p3) +
                               (512.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-122.) * (powr<3>(cosl1p3)) +
                               (74. + (262.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                              (cosl1p1)) *
                             (cosl1p2)) *
                            (cosl1p2)) *
                       ((powr<8>(l1)) * (powr<2>(p))) +
                   (9.) *
                       ((6.) *
                            (-48. + (253.) * (powr<2>(cosl1p1)) + (399.) * (powr<4>(cosl1p1)) +
                             (126.) * (powr<6>(cosl1p1))) *
                            (cosl1p1) +
                        (-112. + (3051.) * (powr<6>(cosl1p1)) + (3942.) * ((powr<5>(cosl1p1)) * (cosl1p3)) +
                         (6.) * (powr<2>(cosl1p3)) + (414.) * (powr<4>(cosl1p3)) +
                         (3.) * (467. + (1089.) * (powr<2>(cosl1p3))) * ((powr<3>(cosl1p1)) * (cosl1p3)) +
                         (5177.999999999999 + (4077.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p1)) +
                         (3.) * (-152. + (874.) * (powr<2>(cosl1p3)) + (27.) * (powr<4>(cosl1p3))) *
                             ((cosl1p1) * (cosl1p3)) +
                         (3.) * (524. + (1049.) * (powr<2>(cosl1p3)) + (342.) * (powr<4>(cosl1p3))) *
                             (powr<2>(cosl1p1))) *
                            (cosl1p3) +
                        (-176. + (2982.) * (powr<2>(cosl1p1)) + (6792.) * (powr<4>(cosl1p1)) +
                         (2241.) * (powr<6>(cosl1p1)) +
                         (3.) *
                             ((2436.) * (powr<5>(cosl1p1)) + (2691.) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                              (9.) * (90. + (181.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p1)) * (cosl1p3)) +
                              (12.) * (237. + (232.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p1)) +
                              (-166. + (546.) * (powr<2>(cosl1p3)) + (9.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                              (6.) * (32. + (273.) * (powr<2>(cosl1p3)) + (54.) * (powr<4>(cosl1p3))) * (cosl1p1)) *
                             (cosl1p3) +
                         (3.) *
                             ((318. + (1543.) * (powr<2>(cosl1p1)) + (504.) * (powr<4>(cosl1p1))) * (cosl1p1) +
                              (3.) *
                                  (-106. + (268.) * (powr<4>(cosl1p1)) + (246.) * ((powr<3>(cosl1p1)) * (cosl1p3)) +
                                   (129.) * (powr<2>(cosl1p3)) + (12.) * (powr<4>(cosl1p3)) +
                                   (cosl1p1) * (-291. + (200.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                                   (-411. + (482.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p1))) *
                                  (cosl1p3) +
                              (-1.) *
                                  (172. + (1224.) * (powr<4>(cosl1p1)) + (36.) * (powr<4>(cosl1p2)) +
                                   (126.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (1014.) * (powr<2>(cosl1p3)) +
                                   (-36.) * (powr<4>(cosl1p3)) +
                                   (27.) * ((101.) * (cosl1p2) + (128.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                                   (15.) * (125. + (3.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
                                   (18.) * (43. + (8.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                                   (2024. + (2052.) * (powr<2>(cosl1p2)) + (4449.) * ((cosl1p2) * (cosl1p3)) +
                                    (2040.) * (powr<2>(cosl1p3))) *
                                       (powr<2>(cosl1p1)) +
                                   ((594.) * (powr<3>(cosl1p2)) + (1752.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                                    (6.) * (759.0000000000001 + (8.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                                    (7.) * (391. + (213.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                                       (cosl1p1)) *
                                  (cosl1p2)) *
                             (cosl1p2)) *
                            (cosl1p2)) *
                       ((powr<7>(l1)) * (powr<3>(p))) +
                   (3.) *
                       (1252. + (-1548.) * (powr<2>(cosl1p1)) + (-16527.) * (powr<4>(cosl1p1)) +
                        (-7776.000000000001) * (powr<6>(cosl1p1)) + (-324.) * (powr<8>(cosl1p1)) +
                        (-3.) *
                            ((486.) * (powr<7>(cosl1p1)) + (756.) * ((powr<6>(cosl1p1)) * (cosl1p3)) +
                             (117.) * (35. + (6.) * (powr<2>(cosl1p3))) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                             (6996. + (810.) * (powr<2>(cosl1p3))) * (powr<5>(cosl1p1)) +
                             (2.) * (98. + (225.) * (powr<2>(cosl1p3)) + (9.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                             (3.) * (402.9999999999999 + (1323.) * (powr<2>(cosl1p3)) + (18.) * (powr<4>(cosl1p3))) *
                                 ((powr<2>(cosl1p1)) * (cosl1p3)) +
                             (9617. + (3660.) * (powr<2>(cosl1p3)) + (324.) * (powr<4>(cosl1p3))) * (powr<3>(cosl1p1)) +
                             (2.) * (552. + (184.) * (powr<2>(cosl1p3)) + (513.) * (powr<4>(cosl1p3))) * (cosl1p1)) *
                            (cosl1p3) +
                        (3.) *
                            ((-378.) * (powr<7>(cosl1p1)) + (232.) * (cosl1p3) +
                             (-1395.) * ((powr<6>(cosl1p1)) * (cosl1p3)) + (-456.) * (powr<3>(cosl1p3)) +
                             (-414.) * (powr<5>(cosl1p3)) +
                             (-15.) * (1045. + (123.) * (powr<2>(cosl1p3))) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                             (-12.) * (713. + (150.) * (powr<2>(cosl1p3))) * (powr<5>(cosl1p1)) +
                             (-1.) * (10039. + (8565.) * (powr<2>(cosl1p3)) + (378.) * (powr<4>(cosl1p3))) *
                                 ((powr<2>(cosl1p1)) * (cosl1p3)) +
                             (-1.) *
                                 (12419. + (7281.000000000001) * (powr<2>(cosl1p3)) + (1305.) * (powr<4>(cosl1p3))) *
                                 (powr<3>(cosl1p1)) +
                             (-1.) *
                                 (-72. + (-1130.) * (powr<2>(cosl1p3)) + (4725.) * (powr<4>(cosl1p3)) +
                                  (27.) * (powr<6>(cosl1p3))) *
                                 (cosl1p1) +
                             (392. + (-5412.000000000001) * (powr<2>(cosl1p1)) + (-7995.) * (powr<4>(cosl1p1)) +
                              (-378.) * (powr<6>(cosl1p1)) +
                              (-1.) *
                                  ((900.) * (powr<5>(cosl1p1)) + (855.) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                                   (9.) * (-275. + (113.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                                   (3.) * (-808. + (231.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p1)) * (cosl1p3)) +
                                   (3.) * (613. + (396.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p1)) +
                                   (-5225. + (4167.) * (powr<2>(cosl1p3)) + (108.) * (powr<4>(cosl1p3))) * (cosl1p1)) *
                                  (cosl1p3) +
                              ((432.) * (powr<5>(cosl1p1)) + (5558.999999999999) * (cosl1p3) +
                               (1620.) * ((powr<4>(cosl1p1)) * (cosl1p3)) + (-180.) * (powr<3>(cosl1p3)) +
                               (36.) * (583. + powr<2>(cosl1p3)) * ((powr<2>(cosl1p1)) * (cosl1p3)) +
                               (72.) * (75. + (19.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p1)) +
                               (4765. + (8328.) * (powr<2>(cosl1p3)) + (-108.) * (powr<4>(cosl1p3))) * (cosl1p1) +
                               (3.) *
                                   (939. + (486.) * (powr<4>(cosl1p1)) + (336.) * (powr<2>(cosl1p2)) +
                                    (1008.) * ((cosl1p2) * (cosl1p3)) + (819.) * (powr<2>(cosl1p3)) +
                                    (3.) * ((162.) * (cosl1p2) + (343.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                                    (4507. + (216.) * (powr<2>(cosl1p2)) + (618.) * ((cosl1p2) * (cosl1p3)) +
                                     (527.9999999999999) * (powr<2>(cosl1p3))) *
                                        (powr<2>(cosl1p1)) +
                                    ((36.) * (powr<3>(cosl1p2)) + (126.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                                     (15.) * (368. + (3.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                                     (4.) * (593. + (36.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                                        (cosl1p1)) *
                                   (cosl1p2)) *
                                  (cosl1p2)) *
                                 (cosl1p2)) *
                            (cosl1p2)) *
                       ((powr<6>(l1)) * (powr<4>(p))) +
                   (3.) *
                       ((1728.) * (powr<7>(cosl1p1)) + (4734.) * ((powr<6>(cosl1p1)) * (cosl1p3)) +
                        (9.) * (5033. + (86.) * (powr<2>(cosl1p3))) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                        (6.) * (3548. + (447.0000000000001) * (powr<2>(cosl1p3))) * (powr<5>(cosl1p1)) +
                        (2.) * (-604. + (201.) * (powr<2>(cosl1p3)) + (342.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                        (2.) * (15017. + (579.) * (powr<2>(cosl1p3)) + (540.) * (powr<4>(cosl1p3))) *
                            ((powr<2>(cosl1p1)) * (cosl1p3)) +
                        (3.) * (5930. + (5097.) * (powr<2>(cosl1p3)) + (708.) * (powr<4>(cosl1p3))) *
                            (powr<3>(cosl1p1)) +
                        (-5656.000000000001 + (7803.999999999999) * (powr<2>(cosl1p3)) + (3918.) * (powr<4>(cosl1p3)) +
                         (54.) * (powr<6>(cosl1p3))) *
                            (cosl1p1) +
                        (-4448. + (23336.) * (powr<2>(cosl1p1)) + (61143.00000000001) * (powr<4>(cosl1p1)) +
                         (7362.) * (powr<6>(cosl1p1)) +
                         (3.) *
                             ((4998.) * (powr<5>(cosl1p1)) + (2283.) * ((powr<4>(cosl1p1)) * (cosl1p3)) +
                              (2.) * (-85. + (444.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                              (3.) * (1067. + (475.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p1)) * (cosl1p3)) +
                              (2.) * (13202. + (770.9999999999999) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p1)) +
                              (2.) * (2636. + (808.) * (powr<2>(cosl1p3)) + (99.) * (powr<4>(cosl1p3))) * (cosl1p1)) *
                             (cosl1p3) +
                         ((1106. + (46983.00000000001) * (powr<2>(cosl1p1)) + (10566.) * (powr<4>(cosl1p1))) *
                              (cosl1p1) +
                          (9.) *
                              (-656. + (1169.) * (powr<4>(cosl1p1)) + (14.) * ((powr<3>(cosl1p1)) * (cosl1p3)) +
                               (52.) * (powr<2>(cosl1p3)) +
                               (4.) * (59. + (111.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p1)) +
                               (cosl1p1) * (-2231. + (195.) * (powr<2>(cosl1p3))) * (cosl1p3)) *
                              (cosl1p3) +
                          (-3.) *
                              ((-702.) * (powr<4>(cosl1p1)) +
                               (6.) * ((464.) * (cosl1p2) + (647.0000000000001) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                               (538. + (738.) * (powr<2>(cosl1p2)) + (1847.) * ((cosl1p2) * (cosl1p3)) +
                                (1241.) * (powr<2>(cosl1p3))) *
                                   (3.) +
                               (4729. + (2418.) * (powr<2>(cosl1p2)) + (5679.) * ((cosl1p2) * (cosl1p3)) +
                                (2886.) * (powr<2>(cosl1p3))) *
                                   (powr<2>(cosl1p1)) +
                               ((576.) * (powr<3>(cosl1p2)) + (1728.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                                (4.) * (4139. + (-45.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                                (9091. + (1377.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                                   (cosl1p1)) *
                              (cosl1p2)) *
                             (cosl1p2)) *
                            (cosl1p2)) *
                       ((powr<5>(l1)) * (powr<5>(p))) +
                   (-1.) *
                       ((-3580. + (10980.) * (powr<6>(cosl1p1)) + (24561.) * ((powr<5>(cosl1p1)) * (cosl1p3)) +
                         (2064.) * (powr<2>(cosl1p3)) + (774.) * (powr<4>(cosl1p3)) +
                         (9.) * (5561. + (1108.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p1)) +
                         ((93246.) * (cosl1p3) + (-4338.) * (powr<3>(cosl1p3))) * (powr<3>(cosl1p1)) +
                         (-3.) * (904. + (-11640.) * (powr<2>(cosl1p3)) + (117.) * (powr<4>(cosl1p3))) *
                             (powr<2>(cosl1p1)) +
                         (3.) * (2741. + (466.) * (powr<2>(cosl1p3)) + (126.) * (powr<4>(cosl1p3))) *
                             ((cosl1p1) * (cosl1p3))) *
                            (2.) +
                        (-3.) *
                            ((-27546.) * (powr<5>(cosl1p1)) + (5933.999999999999) * (powr<3>(cosl1p2)) +
                             (2018.) * (cosl1p3) + (11340.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                             (-576.) * (powr<3>(cosl1p3)) +
                             (-3.) * ((11526.) * (cosl1p2) + (14695.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
                             (-23.) *
                                 (3100. + (527.9999999999999) * (powr<2>(cosl1p2)) + (1245.) * ((cosl1p2) * (cosl1p3)) +
                                  (459.) * (powr<2>(cosl1p3))) *
                                 (powr<3>(cosl1p1)) +
                             (5914. + (5004.) * (powr<2>(cosl1p3))) * (cosl1p2) +
                             (3.) *
                                 ((1870.) * (powr<3>(cosl1p2)) + (-25052.) * (cosl1p3) +
                                  (2725.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (662.) * (powr<3>(cosl1p3)) +
                                  (8.) * (-1541. + (163.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                                 (powr<2>(cosl1p1)) +
                             (9098. + (3402.) * (powr<4>(cosl1p2)) + (8847.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                              (-8372.) * (powr<2>(cosl1p3)) + (-594.) * (powr<4>(cosl1p3)) +
                              (37.) * (200. + (189.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                              (2.) * (-1004. + (602.9999999999999) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3))) *
                                 (cosl1p1)) *
                            (cosl1p2)) *
                       ((powr<4>(l1)) * (powr<6>(p))) +
                   (2.) *
                       (((25038.) * (powr<5>(cosl1p1)) +
                         (9.) * ((8431.) * (cosl1p2) + (5479.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
                         (3.) *
                             (8854. + (25497.) * (powr<2>(cosl1p2)) + (33738.) * ((cosl1p2) * (cosl1p3)) +
                              (7784.999999999999) * (powr<2>(cosl1p3))) *
                             (powr<3>(cosl1p1)) +
                         ((3120.) * (powr<3>(cosl1p2)) + (928.) * (cosl1p3) +
                          (2706.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-312.) * (powr<3>(cosl1p3)) +
                          (4357. + (-591.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (-2.) +
                         ((25929.) * (powr<3>(cosl1p2)) + (47060.) * (cosl1p3) +
                          (53433.00000000001) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-1350.) * (powr<3>(cosl1p3)) +
                          (32626. + (24408.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (powr<2>(cosl1p1)) +
                         ((279.) * (powr<4>(cosl1p2)) + (1962.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                          (-6.) * (-4658. + (201.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
                          (448. + (909.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                          (5285. + (-7441.) * (powr<2>(cosl1p3)) + (216.) * (powr<4>(cosl1p3))) * (-2.)) *
                             (cosl1p1)) *
                            (powr<3>(l1)) +
                        (-1.) *
                            ((-2417. + (24486.) * (powr<4>(cosl1p1)) + (-5683.999999999999) * (powr<2>(cosl1p2)) +
                              (-1599.) * ((cosl1p2) * (cosl1p3)) + (1088.) * (powr<2>(cosl1p3)) +
                              (42.) * ((1361.) * (cosl1p2) + (971.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                              (-797.0000000000001 + (41226.) * (powr<2>(cosl1p2)) + (57621.) * ((cosl1p2) * (cosl1p3)) +
                               (16656.) * (powr<2>(cosl1p3))) *
                                  (powr<2>(cosl1p1)) +
                              ((-7569.) * (cosl1p2) + (8550.) * (powr<3>(cosl1p2)) + (5975.) * (cosl1p3) +
                               (17199.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                               (9009.) * ((cosl1p2) * (powr<2>(cosl1p3))) + (360.) * (powr<3>(cosl1p3))) *
                                  (cosl1p1)) *
                                 (powr<2>(l1)) +
                             (-2.) *
                                 ((5139.) * (powr<3>(cosl1p1)) + (-1504.) * (cosl1p2) + (-209.) * (cosl1p3) +
                                  ((8547.) * (cosl1p2) + (6870.) * (cosl1p3)) * (powr<2>(cosl1p1)) +
                                  (3.) *
                                      (-571. + (1136.) * (powr<2>(cosl1p2)) + (1713.) * ((cosl1p2) * (cosl1p3)) +
                                       (577.) * (powr<2>(cosl1p3))) *
                                      (cosl1p1)) *
                                 ((l1) * (p)) +
                             (512.) * (-1. + (3.) * (powr<2>(cosl1p1)) + (3.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
                                 (powr<2>(p))) *
                            (p)) *
                       (powr<7>(p))) *
                  ((_repl12) *
                   ((powr<-1>(1. + powr<6>(k))) *
                    ((powr<-2>((_repl2) * (_repl4) + (_repl6) * (powr<2>(l1)))) *
                     ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                      ((powr<-1>(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p))) *
                       ((powr<-1>((3.) * (powr<2>(l1)) + (-6.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                  (4.) * (powr<2>(p)))) *
                        ((powr<-1>((_repl4) * (_repl7) +
                                   (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p)) *
                                       (_repl8))) *
                         ((powr<-1>((3.) * ((_repl4) * (_repl9)) +
                                    ((3.) * (powr<2>(l1)) + (-6.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                     (4.) * (powr<2>(p))) *
                                        (_repl10))) *
                          ((powr<-1>((_repl4) * (RB(powr<2>(k),
                                                    powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) +
                                     (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                         (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
                           ((ZA3((0.816496580927726) *
                                 (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
                            (ZA3((0.4714045207910317) *
                                 (sqrt((3.) * (powr<2>(l1)) + (-3.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                                       (5.) * (powr<2>(p))))))))))))))))) +
             ((0.02040816326530612) *
                  ((-50.) * (_repl4 + (-1.) * (_repl5)) * ((_repl2) * (powr<6>(k))) +
                   (_repl1) * (1. + powr<6>(k)) * (_repl2) + (_repl3) * (1. + powr<6>(k)) * (_repl4)) *
                  ((powr<-1>(1. + powr<6>(k))) *
                   ((-3.) *
                        (40. + (9.) * (powr<4>(cosl1p1)) + (-18.) * (powr<4>(cosl1p2)) +
                         (-36.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-21.) * (powr<2>(cosl1p3)) +
                         (9.) * (powr<4>(cosl1p3)) + (9.) * (cosl1p2 + (3.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                         (-3.) *
                             (7. + (3.) * (powr<2>(cosl1p2)) + (-5.) * ((cosl1p2) * (cosl1p3)) +
                              (-6.) * (powr<2>(cosl1p3))) *
                             (powr<2>(cosl1p1)) +
                         (cosl1p2) * (-70. + (9.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                         (-1.) * (70. + (9.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         (-1.) *
                             ((36.) * (powr<3>(cosl1p2)) + (30.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                              (70. + (-15.) * (powr<2>(cosl1p3))) * (cosl1p2) +
                              (-27.) * (-4. + powr<2>(cosl1p3)) * (cosl1p3)) *
                             (cosl1p1)) *
                        (powr<6>(l1)) +
                    (3.) *
                        ((9.) * (powr<5>(cosl1p1)) + (-36.) * (powr<5>(cosl1p2)) +
                         (-90.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
                         (9.) * ((3.) * (cosl1p2) + (4.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
                         (-2.) * (296. + (27.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
                         (-6. + (9.) * (powr<2>(cosl1p2)) + (78.) * ((cosl1p2) * (cosl1p3)) +
                          (45.) * (powr<2>(cosl1p3))) *
                             (powr<3>(cosl1p1)) +
                         ((-888.) * (cosl1p3) + (9.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         (104. + (-6.) * (powr<2>(cosl1p3)) + (9.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                         (208. + (-308.) * (powr<2>(cosl1p3)) + (27.) * (powr<4>(cosl1p3))) * (cosl1p2) +
                         (-1.) *
                             ((54.) * (powr<3>(cosl1p2)) + (42.) * (cosl1p3) + (9.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                              (-45.) * (powr<3>(cosl1p3)) + (308. + (-66.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (powr<2>(cosl1p1)) +
                         (-1.) *
                             (-104. + (90.) * (powr<4>(cosl1p2)) + (132.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                              (42.) * (powr<2>(cosl1p3)) + (-36.) * (powr<4>(cosl1p3)) +
                              (888. + (9.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                              ((664.0000000000001) * (cosl1p3) + (-78.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((powr<5>(l1)) * (p)) +
                    (powr<4>(l1)) *
                        (-1072. + (-27.) * ((powr<5>(cosl1p1)) * (cosl1p2)) + (54.) * (powr<6>(cosl1p2)) +
                         (6372.000000000001) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                         (162.) * ((powr<5>(cosl1p2)) * (cosl1p3)) + (402.) * (powr<2>(cosl1p3)) +
                         (-99.) * (powr<4>(cosl1p3)) +
                         (-9.) * (11. + (6.) * (powr<2>(cosl1p2)) + (12.) * ((cosl1p2) * (cosl1p3))) *
                             (powr<4>(cosl1p1)) +
                         (27.) * (118. + (5.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p2)) +
                         (1576. + (3579.) * (powr<2>(cosl1p3)) + (-54.) * (powr<4>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         ((1576.) * (cosl1p3) + (393.) * (powr<3>(cosl1p3)) + (-27.) * (powr<5>(cosl1p3))) * (cosl1p2) +
                         (-3.) *
                             ((210.) * (cosl1p3) + (51.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                              (-131. + (45.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (powr<3>(cosl1p1)) +
                         (3.) *
                             (134. + (45.) * (powr<4>(cosl1p2)) + (24.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                              (-354.) * (powr<2>(cosl1p3)) +
                              (1193. + (-48.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                              ((203.) * (cosl1p3) + (-45.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                             (powr<2>(cosl1p1)) +
                         ((162.) * (powr<5>(cosl1p2)) + (876.) * (cosl1p3) + (306.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
                          (-630.) * (powr<3>(cosl1p3)) +
                          (36.) * (177. + (2.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
                          (-9.) * (-732. + (17.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                          (1576. + (609.) * (powr<2>(cosl1p3)) + (-108.) * (powr<4>(cosl1p3))) * (cosl1p2)) *
                             (cosl1p1)) *
                        (powr<2>(p)) +
                    (powr<3>(l1)) *
                        ((99.) * ((powr<4>(cosl1p1)) * (cosl1p2)) + (-1164.) * (powr<5>(cosl1p2)) +
                         (-2910.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
                         (-3.) * (98. + (95.) * (powr<2>(cosl1p2)) + (-210.) * ((cosl1p2) * (cosl1p3))) *
                             (powr<3>(cosl1p1)) +
                         (6.) * (234. + (-49.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                         (-57.) * (196. + (5.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                         (-2.) * (3723.999999999999 + (1065.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
                         (2808. + (-4311.999999999999) * (powr<2>(cosl1p3)) + (99.) * (powr<4>(cosl1p3))) * (cosl1p2) +
                         (-1.) *
                             ((2130.) * (powr<3>(cosl1p2)) + (798.) * (cosl1p3) +
                              (153.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                              (4311.999999999999 + (-1062.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (powr<2>(cosl1p1)) +
                         (-1.) *
                             (-1404. + (2910.) * (powr<4>(cosl1p2)) + (3792.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                              (798.) * (powr<2>(cosl1p3)) +
                              (3.) * (3723.999999999999 + (51.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                              ((8456.) * (cosl1p3) + (-630.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                             (cosl1p1)) *
                        (powr<3>(p)) +
                    (2.) *
                        (-682. + (102.) * ((powr<3>(cosl1p1)) * (cosl1p2)) + (2514.) * (powr<4>(cosl1p2)) +
                         (5028.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (203.) * (powr<2>(cosl1p3)) +
                         (203. + (2616.) * (powr<2>(cosl1p2)) + (234.) * ((cosl1p2) * (cosl1p3))) * (powr<2>(cosl1p1)) +
                         (2.) * (454. + (51.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
                         (4.) * (227. + (654.0000000000001) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         ((5028.) * (powr<3>(cosl1p2)) + (382.) * (cosl1p3) +
                          (5160.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                          (908. + (234.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((powr<2>(l1)) * (powr<4>(p))) +
                    (-772.0000000000001) *
                        ((3.) * ((powr<2>(cosl1p1)) * (cosl1p2)) + (6.) * (powr<3>(cosl1p2)) + (-1.) * (cosl1p3) +
                         (9.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                         (-1. + (9.) * (powr<2>(cosl1p2)) + (6.) * ((cosl1p2) * (cosl1p3))) * (cosl1p1) +
                         (-2. + (3.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                        ((l1) * (powr<5>(p))) +
                    (392.) *
                        (-1. + (3.) * ((cosl1p1) * (cosl1p2)) + (3.) * (powr<2>(cosl1p2)) +
                         (3.) * ((cosl1p2) * (cosl1p3))) *
                        (powr<6>(p))) *
                   ((powr<-2>((_repl2) * (_repl4) + (_repl6) * (powr<2>(l1)))) *
                    ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                     ((powr<-1>(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p))) *
                      ((powr<-1>((_repl4) * (_repl7) +
                                 (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p)) *
                                     (_repl8))) *
                       ((powr<-1>((_repl4) *
                                      (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) +
                                  (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                      (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)))))) *
                        ((ZA3((0.816496580927726) *
                              (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))))) *
                         (ZA4((0.7071067811865475) *
                              (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + (2.) * (cosl1p2) + cosl1p3) * ((l1) * (p)) +
                                    (2.) * (powr<2>(p))))))))))))) +
              (-0.04081632653061224) *
                  ((-50.) * (_repl4 + (-1.) * (_repl5)) * ((_repl2) * (powr<6>(k))) +
                   (_repl1) * (1. + powr<6>(k)) * (_repl2) + (_repl3) * (1. + powr<6>(k)) * (_repl4)) *
                  ((_repl12) *
                   ((27.) *
                        (40. + (9.) * (powr<4>(cosl1p1)) + (9.) * (powr<4>(cosl1p2)) +
                         (9.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-70.) * (powr<2>(cosl1p3)) +
                         (-18.) * (powr<4>(cosl1p3)) + (9.) * ((3.) * (cosl1p2) + cosl1p3) * (powr<3>(cosl1p1)) +
                         (3.) *
                             (-7. + (6.) * (powr<2>(cosl1p2)) + (5.) * ((cosl1p2) * (cosl1p3)) +
                              (-3.) * (powr<2>(cosl1p3))) *
                             (powr<2>(cosl1p1)) +
                         (-3.) * (7. + (3.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         (-2.) * (35. + (18.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
                         ((27.) * (powr<3>(cosl1p2)) + (15.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                          (-6.) * (18. + (5.) * (powr<2>(cosl1p3))) * (cosl1p2) +
                          (-2.) * (35. + (18.) * (powr<2>(cosl1p3))) * (cosl1p3)) *
                             (cosl1p1)) *
                        (powr<6>(l1)) +
                    (-9.) *
                        ((81.) * (powr<5>(cosl1p1)) + (81.) * (powr<5>(cosl1p2)) +
                         (135.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
                         (27.) * ((12.) * (cosl1p2) + (5.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
                         (9.) *
                             (-32. + (45.) * (powr<2>(cosl1p2)) + (42.) * ((cosl1p2) * (cosl1p3)) +
                              (-3.) * (powr<2>(cosl1p3))) *
                             (powr<3>(cosl1p1)) +
                         (-9.) * (32. + (3.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
                         (-6.) * (115. + (63.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                         (4.) * (14. + (93.) * (powr<2>(cosl1p3)) + (-27.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                         (-2.) * (-214. + (6.) * (powr<2>(cosl1p3)) + (189.) * (powr<4>(cosl1p3))) * (cosl1p2) +
                         (3.) *
                             ((135.) * (powr<3>(cosl1p2)) + (126.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                              (-2.) * (115. + (63.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                              (-1.) * (688. + (87.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (powr<2>(cosl1p1)) +
                         (427.9999999999999 + (324.) * (powr<4>(cosl1p2)) + (378.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                          (-12.) * (powr<2>(cosl1p3)) + (-378.) * (powr<4>(cosl1p3)) +
                          (-12.) * (206. + (69.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
                          (-3.) * (688. + (87.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2))) *
                             (cosl1p1)) *
                        ((powr<5>(l1)) * (p)) +
                    (9.) *
                        ((54.) * (powr<6>(cosl1p1)) + (54.) * (powr<6>(cosl1p2)) +
                         (135.) * ((powr<5>(cosl1p2)) * (cosl1p3)) +
                         (135.) * ((2.) * (cosl1p2) + cosl1p3) * (powr<5>(cosl1p1)) +
                         (27.) * (-13. + (2.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p2)) +
                         (9.) *
                             (-39. + (54.) * (powr<2>(cosl1p2)) + (58.) * ((cosl1p2) * (cosl1p3)) +
                              (6.) * (powr<2>(cosl1p3))) *
                             (powr<4>(cosl1p1)) +
                         (-9.) * (47. + (30.) * (powr<2>(cosl1p3))) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                         (-473. + (2019.) * (powr<2>(cosl1p3)) + (-459.) * (powr<4>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         ((-2050.) * (cosl1p3) + (3012.) * (powr<3>(cosl1p3)) + (-270.) * (powr<5>(cosl1p3))) *
                             (cosl1p2) +
                         (508. + (-660.9999999999999) * (powr<2>(cosl1p3)) + (537.) * (powr<4>(cosl1p3)) +
                          (-27.) * (powr<6>(cosl1p3))) *
                             (2.) +
                         (9.) *
                             ((60.) * (powr<3>(cosl1p2)) + (81.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                              (-392. + powr<2>(cosl1p3)) * (cosl1p2) +
                              (-1.) * (47. + (30.) * (powr<2>(cosl1p3))) * (cosl1p3)) *
                             (powr<3>(cosl1p1)) +
                         (-473. + (486.) * (powr<4>(cosl1p2)) + (729.0000000000001) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                          (2019.) * (powr<2>(cosl1p3)) + (-459.) * (powr<4>(cosl1p3)) +
                          (-18.) * (355. + (8.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                          (-9.) * (547.0000000000001 + (106.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3))) *
                             (powr<2>(cosl1p1)) +
                         ((270.) * (powr<5>(cosl1p2)) + (-2050.) * (cosl1p3) +
                          (522.0000000000001) * ((powr<4>(cosl1p2)) * (cosl1p3)) + (3012.) * (powr<3>(cosl1p3)) +
                          (-270.) * (powr<5>(cosl1p3)) + (9.) * (-392. + powr<2>(cosl1p3)) * (powr<3>(cosl1p2)) +
                          (-9.) * (547.0000000000001 + (106.) * (powr<2>(cosl1p3))) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                          (-2.) * (776. + (-1344.) * (powr<2>(cosl1p3)) + (477.) * (powr<4>(cosl1p3))) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((powr<4>(l1)) * (powr<2>(p))) +
                    (9.) *
                        ((72.) * (powr<5>(cosl1p1)) + (72.) * (powr<5>(cosl1p2)) +
                         (3.) * ((458.) * (cosl1p2) + (-87.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
                         (-261.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
                         (3.) *
                             (887. + (1266.) * (powr<2>(cosl1p2)) + (524.) * ((cosl1p2) * (cosl1p3)) +
                              (-687.) * (powr<2>(cosl1p3))) *
                             (powr<3>(cosl1p1)) +
                         (-3.) * (-887. + (687.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
                         ((6402.000000000001) * (cosl1p3) + (-2934.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         (-4.) * (160. + (134.) * (powr<2>(cosl1p3)) + (9.) * (powr<4>(cosl1p3))) * (cosl1p3) +
                         (-2.) * (1400. + (-1397.) * (powr<2>(cosl1p3)) + (621.) * (powr<4>(cosl1p3))) * (cosl1p2) +
                         ((3798.) * (powr<3>(cosl1p2)) + (6402.000000000001) * (cosl1p3) +
                          (3702.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-2934.) * (powr<3>(cosl1p3)) +
                          (9865. + (-4173.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (powr<2>(cosl1p1)) +
                         (-2800. + (1374.) * (powr<4>(cosl1p2)) + (1572.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                          (2794.) * (powr<2>(cosl1p3)) + (-1242.) * (powr<4>(cosl1p3)) +
                          (9865. + (-4173.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                          (-28.) * (-511. + (195.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3))) *
                             (cosl1p1)) *
                        ((powr<3>(l1)) * (powr<3>(p))) +
                    (-3.) *
                        (-4572. + (5723.999999999999) * (powr<4>(cosl1p1)) + (5723.999999999999) * (powr<4>(cosl1p2)) +
                         (11559.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (3412.) * (powr<2>(cosl1p3)) +
                         (-1578.) * (powr<4>(cosl1p3)) +
                         (3.) * ((8760.) * (cosl1p2) + (3853.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                         (-488. + (2865.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                         (-488. + (41112.) * (powr<2>(cosl1p2)) + (40491.) * ((cosl1p2) * (cosl1p3)) +
                          (2865.) * (powr<2>(cosl1p3))) *
                             (powr<2>(cosl1p1)) +
                         ((7532.000000000001) * (cosl1p3) + (-4548.) * (powr<3>(cosl1p3))) * (cosl1p2) +
                         ((26280.) * (powr<3>(cosl1p2)) + (7532.000000000001) * (cosl1p3) +
                          (40491.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-4548.) * (powr<3>(cosl1p3)) +
                          (44. + (8160.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((powr<2>(l1)) * (powr<4>(p))) +
                    (2.) *
                        ((10869.) * (powr<3>(cosl1p1)) + (10869.) * (powr<3>(cosl1p2)) +
                         (18015.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                         (3.) * ((11319.) * (cosl1p2) + (6005.) * (cosl1p3)) * (powr<2>(cosl1p1)) +
                         (-2.) * (777.9999999999999 + (147.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                         (-9026. + (6852.000000000001) * (powr<2>(cosl1p3))) * (cosl1p2) +
                         (-9026. + (33957.) * (powr<2>(cosl1p2)) + (37380.) * ((cosl1p2) * (cosl1p3)) +
                          (6852.000000000001) * (powr<2>(cosl1p3))) *
                             (cosl1p1)) *
                        ((l1) * (powr<5>(p))) +
                    (-8.) *
                        (-584. + (843.) * (powr<2>(cosl1p1)) + (1686.) * ((cosl1p1) * (cosl1p2)) +
                         (843.) * (powr<2>(cosl1p2)) + (909.) * ((cosl1p1) * (cosl1p3)) +
                         (909.) * ((cosl1p2) * (cosl1p3)) + (66.) * (powr<2>(cosl1p3))) *
                        (powr<6>(p))) *
                   ((powr<-1>(1. + powr<6>(k))) *
                    ((powr<-2>((_repl2) * (_repl4) + (_repl6) * (powr<2>(l1)))) *
                     ((powr<-1>(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p))) *
                      ((powr<-1>((3.) * (powr<2>(l1)) + (-6.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                 (4.) * (powr<2>(p)))) *
                       ((powr<-1>((_repl4) * (_repl7) +
                                  (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p)) *
                                      (_repl8))) *
                        ((powr<-1>(
                             (3.) * ((_repl4) * (_repl9)) +
                             ((3.) * (powr<2>(l1)) + (-6.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (4.) * (powr<2>(p))) *
                                 (_repl10))) *
                         (ZA4((0.408248290463863) *
                              (sqrt((3.) * (powr<2>(l1)) + (-3.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                    (5.) * (powr<2>(p)))))))))))))) *
                 (_repl11) +
             (0.01530612244897959) *
                 ((-50.) * (_repl4 + (-1.) * (_repl5)) * ((_repl2) * (powr<6>(k))) +
                  (_repl1) * (1. + powr<6>(k)) * (_repl2) + (_repl3) * (1. + powr<6>(k)) * (_repl4)) *
                 ((powr<-1>(1. + powr<6>(k))) *
                  ((-3.) *
                       (-1304. + (54.) * (powr<4>(cosl1p1)) + (-27.) * (powr<4>(cosl1p2)) +
                        (-81.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-399.) * (powr<2>(cosl1p3)) +
                        (-27.) * (powr<4>(cosl1p3)) + (108.) * (cosl1p2 + cosl1p3) * (powr<3>(cosl1p1)) +
                        (3.) *
                            (-262. + (9.) * (powr<2>(cosl1p2)) + (30.) * ((cosl1p2) * (cosl1p3)) +
                             (9.) * (powr<2>(cosl1p3))) *
                            (powr<2>(cosl1p1)) +
                        (-3.) * (133. + (18.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                        (-3.) * (4. + (27.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
                        (-3.) *
                            ((262.) * (cosl1p2) + (9.) * (powr<3>(cosl1p2)) + (262.) * (cosl1p3) +
                             (15.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (15.) * ((cosl1p2) * (powr<2>(cosl1p3))) +
                             (9.) * (powr<3>(cosl1p3))) *
                            (cosl1p1)) *
                       (powr<2>(l1)) +
                   (-9.) * (cosl1p2 + cosl1p3) *
                       ((l1) *
                        (872. + (282.) * (powr<2>(cosl1p1)) + (151.) * (powr<2>(cosl1p2)) +
                         (20.) * ((cosl1p2) * (cosl1p3)) + (151.) * (powr<2>(cosl1p3)) +
                         (282.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
                        (p)) +
                   (16.) *
                       (382. + (57.) * (powr<2>(cosl1p1)) + (-9.) * (powr<2>(cosl1p2)) +
                        (-75.) * ((cosl1p2) * (cosl1p3)) + (-9.) * (powr<2>(cosl1p3)) +
                        (57.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
                       (powr<2>(p))) *
                  ((powr<-2>((_repl2) * (_repl4) + (_repl6) * (powr<2>(l1)))) *
                   ((powr<-1>((3.) * (powr<2>(l1)) + (-6.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                              (4.) * (powr<2>(p)))) *
                    ((powr<-1>(
                         (3.) * ((_repl4) * (RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                                                (1.333333333333333) * (powr<2>(p))))) +
                         ((3.) * (powr<2>(l1)) + (-6.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) + (4.) * (powr<2>(p))) *
                             (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                      (1.333333333333333) * (powr<2>(p))))))) *
                     (powr<2>(ZA4((0.408248290463863) *
                                  (sqrt((3.) * (powr<2>(l1)) + (-3.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                        (5.) * (powr<2>(p))))))))))) +
             (0.06122448979591837) *
                 ((9.) *
                      ((3.) * (powr<4>(cosl1p1)) + (-6.) * (powr<4>(cosl1p2)) +
                       (-12.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
                       (3.) * (cosl1p2 + (3.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
                       (2. + (-3.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                       (-1. + (3.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p3)) +
                       (cosl1p2) * (2. + (3.) * (powr<2>(cosl1p3))) * (cosl1p3) +
                       (-1. + (-3.) * (powr<2>(cosl1p2)) + (5.) * ((cosl1p2) * (cosl1p3)) + (6.) * (powr<2>(cosl1p3))) *
                           (powr<2>(cosl1p1)) +
                       ((2.) * (cosl1p2) + (-12.) * (powr<3>(cosl1p2)) + (-4.) * (cosl1p3) +
                        (-10.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (5.) * ((cosl1p2) * (powr<2>(cosl1p3))) +
                        (9.) * (powr<3>(cosl1p3))) *
                           (cosl1p1)) *
                      (powr<2>(l1)) +
                  (3.) *
                      ((12.) * (powr<3>(cosl1p1)) + (-18.) * (powr<3>(cosl1p2)) + (-4.) * (cosl1p3) +
                       (-21.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (6.) * (powr<3>(cosl1p3)) +
                       ((17.) * (cosl1p2) + (19.) * (cosl1p3)) * (powr<2>(cosl1p1)) +
                       (4. + (-3.) * (powr<2>(cosl1p3))) * (cosl1p2) +
                       (-1.) * ((7.) * (powr<2>(cosl1p2)) + (5.) * (powr<2>(cosl1p3))) * (cosl1p1)) *
                      ((l1) * (p)) +
                  (-16. + (43.) * (powr<2>(cosl1p1)) + (52.) * ((cosl1p1) * (cosl1p2)) + (9.) * (powr<2>(cosl1p2)) +
                   (34.) * ((cosl1p1) * (cosl1p3)) + (6.) * ((cosl1p2) * (cosl1p3))) *
                      (powr<2>(p))) *
                 ((powr<2>(l1)) *
                  ((_repl13) * (_repl3) + (_repl2) * (dtZc(k)) +
                   (-50.) * (_repl13 + (-1.) * (Zc((1.02) * (k)))) * (_repl2)) *
                  ((ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))))) *
                   ((ZAcbc((0.816496580927726) *
                           (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2 + cosl1p3) * (p) + powr<2>(p))))) *
                    ((ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (l1) * ((2.) * (cosl1p1) + cosl1p2) * (p) +
                                                        (1.666666666666667) * (powr<2>(p)))))) *
                     ((ZAcbc((0.816496580927726) *
                             (sqrt(powr<2>(l1) + (l1) * ((2.) * (cosl1p1) + (2.) * (cosl1p2) + cosl1p3) * (p) +
                                   (1.666666666666667) * (powr<2>(p)))))) *
                      ((powr<-2>((_repl13) * (_repl2) + (powr<2>(l1)) * (Zc(l1)))) *
                       ((powr<-1>((_repl13) *
                                      (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) +
                                  (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                      (Zc(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
                        ((powr<-1>((_repl13) * (RB(powr<2>(k), powr<2>(l1) +
                                                                   (2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                                                   powr<2>(p))) +
                                   (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p)) *
                                       (Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                                powr<2>(p)))))) *
                         (powr<-1>(
                             (3.) *
                                 ((_repl13) * (RB(powr<2>(k), powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                                                  (1.333333333333333) * (powr<2>(p))))) +
                             ((3.) * (powr<2>(l1)) + (6.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (4.) * (powr<2>(p))) *
                                 (Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                          (1.333333333333333) * (powr<2>(p)))))))))))))));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto
    constant(const auto &p, const double &k,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZAcbc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_FORCEINLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return Regulator::RBdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return Regulator::RFdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return Regulator::dq2RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return Regulator::dq2RF(k2, p2); }
  };
} // namespace DiFfRG
using DiFfRG::ZA4_kernel;