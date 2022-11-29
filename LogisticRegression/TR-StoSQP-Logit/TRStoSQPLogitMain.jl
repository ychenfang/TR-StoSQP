include("TRStoSQPLogit.jl")
struct TRStoSQPLogitResult
    KKTIdC::Array       # KKT residual of constant beta_k with identity Hessian
    KKTSR1C::Array      # KKT residual of constant beta_k with SR1 update Hessian
    KKTEstHC::Array     # KKT residual of constant beta_k with estimated Hessian
    KKTAveHC::Array     # KKT residual of constant beta_k with averaged estimated Hessian
    KKTIdD::Array       # KKT residual of decay beta_k with identity Hessian
    KKTSR1D::Array      # KKT residual of decay beta_k with SR1 update Hessian
    KKTEstHD::Array     # KKT residual of decay beta_k with estimated Hessian
    KKTAveHD::Array     # KKT residual of decay beta_k with averaged estimated Hessian
end

## Implement Fully Stochastic TR-SQP for whole problem set
# FullyTR: parameters of Fully Stochastic algorithm
# Prob: problem name set

function TRStoSQPLogitMain(TRSto, data)
    Verbose = TRSto.verbose
    BetaCSet = TRSto.constbeta
    BetaDSet = TRSto.decaybeta
    zeta = TRSto.zeta
    mu_1 = TRSto.mu_1
    rho = TRSto.rho
    Max_Epoch = TRSto.MaxEpoch
    TotalRep = TRSto.Rep
    EPS_Res = TRSto.EPS_Res
    BatchSize = TRSto.BatchSize
    LenCStep = length(BetaCSet)
    LenDStep = length(BetaDSet)
    LenBatchSize = length(BatchSize)

    TRStoSQPLogitR = Array{TRStoSQPLogitResult}(undef,1)
    # load problem
    n = size(data,1)
    y = data[1:n-5,1]
    Z = data[1:n-5,2:end]
    b = data[n-4:n,1]
    A = data[n-4:n,2:end]

    # define results vector for constant stepsize
    KKTCStep = reshape([[] for i=1:LenCStep for j=1:LenBatchSize], LenCStep,:)
    # go over constant stepsize
    i = 1
    while i <= LenCStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP Const-Id","-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaCSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 1, 0)
                println(KKT,"-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTCStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    # define results vector for constant stepsize
    KKTSR1CStep = reshape([[] for i=1:LenCStep for j=1:LenBatchSize], LenCStep,:)

    # go over constant stepsize
    i = 1
    while i <= LenCStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP Const-SR1","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaCSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res,1,1)
                println(KKT,"-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTSR1CStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    # define results vector for constant stepsize
    KKTEstHCStep = reshape([[] for i=1:LenCStep for j=1:LenBatchSize], LenCStep,:)

    # go over constant stepsize
    i = 1
    while i <= LenCStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP Const-EstH","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaCSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 1, 2)
                println(KKT,"-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTEstHCStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    # define results vector for constant stepsize
    KKTAveHCStep = reshape([[] for i=1:LenCStep for j=1:LenBatchSize], LenCStep,:)

    # go over constant stepsize
    i = 1
    while i <= LenCStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP Const-AveH","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaCSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 1, 3)
                println(KKT,"-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTAveHCStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end



    # define results vector for decaying stepsize
    KKTDStep = reshape([[] for i=1:LenCStep for j=1:LenBatchSize], LenCStep,:)

    # go over decay stepsize
    i = 1
    while i <= LenDStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP decay-Id","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaDSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 0, 0)
                println(KKT, "-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTDStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    # define results vector for decaying stepsize
    KKTSR1DStep = reshape([[] for i=1:LenDStep for j=1:LenBatchSize], LenDStep,:)

    # go over decay stepsize
    i = 1
    while i <= LenDStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP decay-SR1","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaDSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 0, 1)
                println(KKT, "-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTSR1DStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    # define results vector for decaying stepsize
    KKTEstHDStep = reshape([[] for i=1:LenDStep for j=1:LenBatchSize], LenDStep,:)

    # go over decay stepsize
    i = 1
    while i <= LenDStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP decay-EstH","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaDSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 0, 2)
                println(KKT, "-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTEstHDStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    # define results vector for decaying stepsize
    KKTAveHDStep = reshape([[] for i=1:LenDStep for j=1:LenBatchSize], LenDStep,:)

    # go over decay stepsize
    i = 1
    while i <= LenDStep
        j = 1
        while j <= LenBatchSize
            rep = 1
            while rep <= TotalRep
                println("TRStoSQP decay-AveH","-",Idprob,"-",i,"-",j,"-",rep)
                KKT, IdCon, IdSing = TRStoSQPLogit(Z, y, A, b, BetaDSet[i], zeta, mu_1, rho, BatchSize[j], Max_Epoch, EPS_Res, 0, 3)
                println(KKT, "-", IdCon,"-",IdSing)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(KKTAveHDStep[i, j], KKT)
                    rep += 1
                end
            end
            j += 1
        end
        i += 1
    end

    TRStoSQPLogitR = TRStoSQPLogitResult(KKTCStep, KKTSR1CStep, KKTEstHCStep, KKTAveHCStep, KKTDStep, KKTSR1DStep, KKTEstHDStep, KKTAveHDStep)

    return TRStoSQPLogitR
end
