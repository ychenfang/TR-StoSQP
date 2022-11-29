
## Load packages
using NLPModels
using JuMP
using LinearOperators
using OptimizationProblems
using MathProgBase
using ForwardDiff
using CUTEst
using NLPModelsJuMP
using LinearAlgebra
using Distributed
using Ipopt
using DataFrames
using PyPlot
using MATLAB
using Glob
using DelimitedFiles
using Random
using Distributions
using Ipopt

cd("/.../TRStoSQP")


# define parameter module
module Parameter
    # Parameters of TR-StoSQP with l2 merit function and l1-StoSQP (Berahas method)

    struct TRStoParams
        verbose
        constbeta::Array{Float64}          # constant radius-related sequence
        decaybeta::Array{Float64}          # decay radius-related sequence 1/(k^p),0.5<p<1
        zeta::Float64                      # zeta
        mu_1::Float64                      # mu_{-1}
        rho::Float64                       # rho
        MaxIter::Int                       # Maximum Iteration
        Rep::Int                           # Number of Independent runs
        EPS_Res::Float64                   # minimum of difference
        Sigma::Array{Float64}              # variance of gradient
    end


end

using Main.Parameter


include("TRStoSQPMain.jl")
Prob = readdlm(string(pwd(),"/../Parameter/Fully_problems.txt"))
#######################################
#########  run main file    ###########
#######################################
function main()
    ## run FullyTRSQP
    include("../Parameter/ParamsTRStoSQP.jl")
    TRStoResult = TRStoSQPMain(TRSto, Prob)
    return TRStoResult
end

main()
