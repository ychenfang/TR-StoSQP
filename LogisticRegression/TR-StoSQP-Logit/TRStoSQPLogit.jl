## Implement TR-StoSQP for a single problem of Logistic regression
# Input
### nlp: problem
### Beta: beta_k
### sigma: standard deviation of approximation
### Max_Iter: maximum iteration
### EPS_Res: Stopping criterion of KKT residual
### IdConst: indication of constant beta_k
###    IdConst == 1: constant beta_k
###    IdConst == 0: decay beta_k
### IdB_k: Indicator of Hessian approximation used,
###    0 for Identity; 1 for SR1 update, 2 for estimated Hessian, 3 for averaged estimated Hessiann

# Output
### KKT: KKT residual iteration sequence
### IdCon: indicator of whether convergence
### IdSing: indicator of singularity


include("sr1.jl")
include("gradient.jl")
include("hessian.jl")
include("objective.jl")
function TRStoSQPLogit(Z, y, A, b, beta, zeta, mu_1, rho, batchsize, Max_Epoch, EPS_Res, IdConst, IdB_k)

    nx = size(Z,2)
    sample_size = size(Z,1)
    Max_Iter = Max_Epoch * sample_size/batchsize
    nlam = size(A,1)            # number of constraints
    IdCon, IdSing = 1, 0
    mu = mu_1

    # Initialize
    eps, k, X = 1, 0, [ones(nx)]
    # number of three cases:
    n1,n2,n3 =0,0,0
    # storing B_k and bnab_xL_k
    # storing SR1 hessian
    B = []
    push!(B,Diagonal(ones(nx)))
    push!(B,Diagonal(ones(nx)))
    bnab_xL = []
    # storing estimated hessian
    H = []
    push!(H, Diagonal(ones(nx)))

    # evaluate constraint and Jacobian
    c_k = A*X[end]-b
    G_k = A
    nabf_k = grad_logit(Z,y,X[end])
    P0 = Matrix(G_k*G_k')
    try
        invP = inv(Matrix(G_k*G_k'))
        P0 = invP
    catch
        IdSing = 1
    end
    if IdSing == 1
        return [], IdCon, IdSing
    else
        lam_k = P0*(-G_k*nabf_k)
        nab_xL_k = nabf_k + G_k'lam_k
        KKT = [norm([nab_xL_k; c_k])]
        nab_x2L_k = hess_logit(Z,y,X[end])
        IdSetup = 1
    end

    # estimate Lipschitz constants Lipf and LipG
    eps_x = 0.1*ones(nx)
    x_trial = X[end]+eps_x
    # Lipf:
    f_k = obj_logit(Z,y,X[end])
    f_trial = obj_logit(Z,y,x_trial)
    c_trial = A*x_trial-b
    G_trial = A
    LipG = norm(G_trial-G_k)/norm(eps_x)+1
    Lipf = norm(f_trial-f_k)/norm(eps_x)+1

        while KKT[end]>EPS_Res && k<Max_Iter
            println(KKT[end])
        ## Obtain est of gradient of Lagrange
        # stochastic gradient:
            # select a subset:
            idx = rand(1:sample_size,batchsize)
            Z_batch = Z[idx,:]
            y_batch = y[idx]
            # stochastic gradient and Lagrangian multiplier:
            bnabf_k = grad_logit(Z_batch,y_batch,X[end])
            blam_k = P0 * (-G_k*bnabf_k)
            bnab_xL_k = bnabf_k+G_k'*blam_k
            push!(bnab_xL,bnab_xL_k)
            bnab_x2L_k = hess_logit(Z_batch,y_batch,X[end])
            push!(H,bnab_x2L_k)
            bKKT = norm([bnab_xL_k; c_k])
            beta_k = IdConst * beta + (1-IdConst)*(1/(k+1))^beta
            beta_max = IdConst * beta + (1-IdConst)*(1/(0+1))^beta
            ## generate B_k:
            if IdB_k == 0
                B_k = Diagonal(ones(nx))
            elseif IdB_k == 1
                if k>=1
                    H_k = sr1(B,bnab_xL,X)
                    push!(B,H_k)
                end
                B_k = B[end-1]
            elseif IdB_k == 2
                B_k = H[end-1]
            else
                if k<=100
                    B_k = mean(H[1:end-1])
                else
                    B_k = mean(H[end-100:end-1])
                end
            end

            # compute control parameters
            eta1 = zeta * min(1/norm(B_k), 6*beta_max/norm(G_k))
            tau_k = Lipf + LipG * mu + norm(B_k)
            alpha_k = beta_k/((4*eta1*tau_k+6*zeta)*beta_max)
            eta2 = eta1 - 1/2 * zeta * eta1 * alpha_k

            ## Generate trust region radius:
            if bKKT < 1/eta1
                n1 = n1+1
                delta_k = eta1*alpha_k*bKKT
            elseif bKKT <= 1/eta2
                n2 = n2+1
                delta_k = alpha_k
            else
                n3 = n3+1
                delta_k = eta2*alpha_k*bKKT
            end

        ## Decompose trust region radius:
        ## check_delta_k:
            ck_delta_k = norm(c_k)/bKKT*delta_k
        ## tilde_delta_k:
            td_delta_k = norm(bnab_xL_k)/bKKT*delta_k
        ## Compute \bv_k:
            v_k = -G_k'* P0*c_k
        ## Select \gamma_k:
            gamma_k = min(ck_delta_k/norm(v_k),1)
        # Compute P_k:
            P_k = Diagonal(ones(nx))-G_k'* P0*G_k

        ## Compute \bu_k:
            m = Model(Ipopt.Optimizer);
            set_silent(m)
            @variable(m, u[1:nx]);
            @objective(m, Min, bnabf_k' * P_k * u + 1/2 * u' * P_k * B_k * P_k * u);
            @constraint(m,u'*u <= td_delta_k^2)
            optimize!(m);
            u_k=Float64[];
            for i=1:nx
              push!(u_k, callback_value(m,u[i]))
            end

        ## Compute the trial step:
            deltax_k = gamma_k*v_k+P_k*u_k

        # compute predicted reduction:
            pred_k = bnabf_k'*deltax_k + 1/2*deltax_k'*B_k*deltax_k + mu*(norm(c_k + G_k*deltax_k)-norm(c_k))
            lower_bd_pred_k = -norm(bnab_xL_k)*td_delta_k - 1/2*norm(c_k)*ck_delta_k + 1/2*td_delta_k^2*norm(B_k) + norm(B_k)*td_delta_k*ck_delta_k
            # update mu
            while pred_k > lower_bd_pred_k
                if mu > 1e4
                    break
                else
                    mu *= rho
                    pred_k = bnabf_k'*deltax_k + 1/2*deltax_k'*B_k*deltax_k + mu*(norm(c_k + G_k*deltax_k)-norm(c_k))
                end
            end
        # compute inverse, if singluar, quit
            push!(X, X[end] + deltax_k)
            eps = norm(deltax_k)
            k = k+1
            c_k = A*X[end]-b
            G_k = A
            nabf_k = grad_logit(Z,y,X[end])
            P0 = Matrix(G_k*G_k')
            try
                invP = inv(Matrix(G_k*G_k'))
                P0=invP
            catch
                IdSing = 1
            end
            if IdSing == 1
                return [], IdCon, IdSing
            else
                lam_k= -P0*G_k*nabf_k
                nab_xL_k = nabf_k + G_k'lam_k
                push!(KKT, norm([nab_xL_k; c_k]))
            end
        end

    s = n1+n2+n3
    Por = []
    push!(Por,n1/s)
    push!(Por,n2/s)
    push!(Por,n3/s)

    if k < Max_Iter
        return KKT[end], 1, 0
    else
        return [], 0, 1
    end
end
