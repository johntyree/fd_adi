

from VecArray cimport SizedArray

cdef extern from "_coefficients.cuh":
    void scale_0(
        double t,
        double r,
        SizedArray[double] &spots,
        SizedArray[double] &vars,
        SizedArray[double] &vec) except +

    void scale_00(
        double t,
        double r,
        SizedArray[double] &spots,
        SizedArray[double] &vars,
        SizedArray[double] &vec) except +

    void scale_1(
        double t,
        double r,
        SizedArray[double] &spots,
        SizedArray[double] &vars,
        double reversion,
        double mean_variance,
        SizedArray[double] &vec) except +

    void scale_11(
        double t,
        double r,
        SizedArray[double] &spots,
        SizedArray[double] &vars,
        double vol_of_var,
        SizedArray[double] &vec) except +

    void scale_01(
        double t,
        double r,
        SizedArray[double] &spots,
        SizedArray[double] &vars,
        double vol_of_var,
        double correlation,
        SizedArray[double] &vec) except +
