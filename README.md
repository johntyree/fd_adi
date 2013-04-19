Finite Differencing on GPUs
===========================

PDE based option pricing on GPUs

Many models used in finance end up in formulation of highly mathematical
problems. Solving these equations exactly in closed form is impossible as the
experience in other fields suggests. Therefore, we have to look for efficient
numerical algorithms in solving complex problems such as option pricing, risk
analysis, portfolio management, etc. Computational finance, generally referring
to the application of computational techniques to finance, has become an
integral part of modeling, analysis, and decision-making in the financial
industry. 

In the world of derivatives pricing there are two main working horses, namely:
Monte Carlo methods and numerical PDE (Partial Differential Equation) based
techniques. In the past few years, we have seen an increasing interest in the
computational finance community for the application of Graphical Processing
Units (GPUs). However, so far this technology has shown most promising results
for Monte Carlo based approaches, while limited analysis has been done on
Finite-Difference based calculations.

In this project we will explore the potentials for the application of GPUs for
PDE based derivatives pricing.


QuickStart
==========

    $ git clone <REPO>
    $ sh autotest.sh  # builds and runs tests

Now price a test option with default parameters using the CPU, the GPU, and
Monte Carlo integration.

    $ python price_one_option.py -nx 150 150 -nt 150 --cpu --gpu --mc 10000 -v
