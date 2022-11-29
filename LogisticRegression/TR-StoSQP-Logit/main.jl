# Loading packages
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

cd("/.../TR-StoSQP-Logit")


# define parameter module
module Parameter
    # Parameters of adaptive LS-SQP with l2 merit function

    struct TRStoParams
        verbose
        constbeta::Array{Float64}          # constant radius-related sequence
        decaybeta::Array{Float64}          # decay radius-related sequence 1/(k^p),0.5<p<1
        zeta::Float64                      # zeta
        mu_1::Float64                      # mu_{-1}
        rho::Float64                       # rho
        MaxEpoch::Int                      # Maximum Epoch
        Rep::Int                           # Number of Independent runs
        EPS_Res::Float64                   # minimum of difference
        BatchSize::Array{Int64}              # variance of gradient
    end


end


using Main.Parameter
include("TRStoSQPLogitMain.jl")
data = readdlm("/.../LogitData/austrilian.txt");

#######################################
#########  run main file    ###########
#######################################
function main()

    ## run TR-StoSQP
    TRStoSQPLogitR = Array{TRStoSQPLogitResult}(undef,1)
    include("/.../Parameter/ParamsLogit.jl")
    TRStoSQPR = TRStoSQPLogitMain(TRSto, data)
    TRStoSQPR = TRStoSQPLogitR

end

main()
