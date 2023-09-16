include("TRStoSQP.jl")
struct TRStoResult
    KKTIdC::Array           # constant beta_k: KKT residual of the last iteration with identity Hessian approximation
    PorIdC::Array           # constant beta_k: Portion of three cases in generating trust-region radius with identity Hessian approximation
    KKTSR1C::Array          # constant beta_k: KKT residual of the last iteration with SR1 update Hessian approximation
    PorSR1C::Array          # constant beta_k: Portion of three cases in generating trust-region radius with SR1 update Hessian approximation
    KKTEstHC::Array         # constant beta_k: KKT residual of the last iteration with estimated Hessian approximation
    PorEstHC::Array         # constant beta_k: Portion of three cases in generating trust-region radius with estimated Hessian approximation
    KKTAveHC::Array         # constant beta_k: KKT residual of the last iteration with averaged estimated Hessian approximation
    PorAveHC::Array         # constant beta_k: Portion of three cases in generating trust-region radius with averaged estimated Hessian approximation
    KKTIdD::Array           # decay beta_k: KKT residual of the last iteration with identity Hessian approximation
    PorIdD::Array           # decay beta_k: Portion of three cases in generating trust-region radius with identity Hessian approximation
    KKTSR1D::Array          # decay beta_k: KKT residual of the last iteration with SR1 update Hessian approximation
    PorSR1D::Array          # decay beta_k: Portion of three cases in generating trust-region radius with SR1 update Hessian approximation
    KKTEstHD::Array         # decay beta_k: KKT residual of the last iteration with estimated Hessian approximation
    PorEstHD::Array         # decay beta_k: Portion of three cases in generating trust-region radius with estimated Hessian approximation
    KKTAveHD::Array         # decay beta_k: KKT residual of the last iteration with averaged estimated Hessian approximation
    PorAveHD::Array         # decay beta_k: Portion of three cases in generating trust-region radius with averaged estimated Hessian approximation
end

## Implement TR-StoSQP for whole problem set
# FullyTR: parameters of Fully Stochastic algorithm
# Prob: problem name set

function TRStoSQPMain(TRSto, Prob)
    Verbose = TRSto.verbose
    BetaCSet = TRSto.constbeta
    BetaDSet = TRSto.decaybeta
    zeta = TRSto.zeta
    ddelta = TR.Sto.ddelta
    mu_1 = TRSto.mu_1
    rho = TRSto.rho
    Max_Iter = TRSto.MaxIter
    TotalRep = TRSto.Rep
    EPS_Res = TRSto.EPS_Res
    Sigma = TRSto.Sigma
    LenCStep = length(BetaCSet)       # constant beta
    LenDStep = length(BetaDSet)       # decay beta
    LenSigma = length(Sigma)

    TRStoR = Array{TRStoResult}(undef,length(Prob))

    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        IdSing = 0
        nlp = CUTEstModel(Prob[Idprob])

        # define results vector for constant beta_k with SR1 update Hessian approximation
        KKTSR1CStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        PorSR1CStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)

        # go over constant beta_k
        i = 1
        while i <= LenCStep
            j = 1
            while j <= LenSigma
               rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Const-SR1","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaCSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res,1,1)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTSR1CStep[i, j], KKT)
                        push!(PorSR1CStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end


        # define results vector for constant beta_k with identity Hessian approximation
        KKTCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        PorCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)

        # go over constant beta_k
        i = 1
        while i <= LenCStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Const-Id","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaCSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 1, 0)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTCStep[i, j], KKT)
                        push!(PorCStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end



        # define results vector for constant beta_k with estimated Hessian approximation
        KKTEstHCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        PorEstHCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)

        # go over constant beta_k
        i = 1
        while i <= LenCStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Const-EstH","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaCSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 1, 2)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTEstHCStep[i, j], KKT)
                        push!(PorEstHCStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end

        # define results vector for constant beta_k with averaged estimated Hessian approximation
        KKTAveHCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        PorAveHCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)

        # go over constant beta_k
        i = 1
        while i <= LenCStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Const-AveH","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaCSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 1, 3)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTAveHCStep[i, j], KKT)
                        push!(PorAveHCStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end


        # define results vector for decay beta_k with identity Hessian approximation
        KKTDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        PorDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)

        # go over decay beta_k
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Decay-Id","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaDSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 0, 0)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTDStep[i, j], KKT)
                        push!(PorDStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end

        # define results vector for decay beta_k with SR1 update Hessian approximation
        KKTSR1DStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        PorSR1DStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)

        # go over decay beta_k
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Decay-SR1","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaDSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 0, 1)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTSR1DStep[i, j], KKT)
                        push!(PorSR1DStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end

        # define results vector for decay beta_k with estimaed Hessian approximation
        KKTEstHDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        PorEstHDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)

        # go over decay beta_k
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Decay-EstH","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaDSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 0, 2)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTEstHDStep[i, j], KKT)
                        push!(PorEstHDStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end

        # define results vector for decay beta_k with averaged estimaed Hessian approximation
        KKTAveHDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        PorAveHDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)

        # go over decay stepsize
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRStoSQP Decay-AveH","-",Idprob,"-",i,"-",j,"-",rep)
                    KKT, Portion, IdCon, IdSing = TRStoSQP(nlp, BetaDSet[i], zeta, ddelta, mu_1, rho, Sigma[j], Max_Iter, EPS_Res, 0, 3)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(KKTAveHDStep[i, j], KKT)
                        push!(PorAveHDStep[i, j], Portion)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end
        TRStoR[Idprob] = TRStoResult(KKTCStep, PorCStep, KKTSR1CStep, PorSR1CStep, KKTEstHCStep, PorEstHCStep, KKTAveHCStep, PorAveHCStep, KKTDStep, PorDStep, KKTSR1DStep, PorSR1DStep, KKTEstHDStep, PorEstHDStep, KKTAveHDStep, PorAveHDStep)
        finalize(nlp)
    end
    return TRStoR
end
