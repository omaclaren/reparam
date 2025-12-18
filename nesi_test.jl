using Pkg
Pkg.activate(".")
Pkg.instantiate()

println("Environment instantiated.")

using Distributed
using Dates

println("Running on NeSI at $(now())")
println("Number of threads: $(Threads.nthreads())")
println("Number of workers: $(nworkers())")

using Distributions
using ForwardDiff
using NLopt
using DifferentialEquations

println("All key packages loaded successfully.")
println("Ready for Repressilator analysis.")
