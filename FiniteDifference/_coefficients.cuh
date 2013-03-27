#ifndef _COEFFICIENTS_CUH
#define _COEFFICIENTS_CUH

#include "VecArray.h"

void scale_0(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , SizedArray<double> &vec
        );

void scale_00(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , SizedArray<double> &vec
        );

void scale_1(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , double reversion
        , double mean_variance
        , SizedArray<double> &vec
        );

void scale_11(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , double vol_of_var
        , SizedArray<double> &vec
        );


void scale_01(
          double t
        , double r
        , SizedArray<double> &spots
        , SizedArray<double> &vars
        , double vol_of_var
        , double correlation
        , SizedArray<double> &vec
        );


#endif /* end of include guard */
