## Implement Fully Stochastic TR SQP for a single problem
# Input
### nlp: problem
### Step: beta_k
### eta1k: eta_{1,k}
### eta2k: eta_{2,k}
### sigma: standard deviation of approximation
### Max_Iter: maximum iteration
### EPS: minimum of difference
### IdConst: indication of constant beta_k
###         IdConst == 1: constant beta_k
###         IdConst == 0: decay beta_k
### IdB_k: Indicator of Hessian approximation used, 0 for Identity,
###        IdB_k == 0 for identity Hessian; IdB_k == 1 for SR1 update; IdB_k == 2 for estimated Hessian; else for averaged estimaed Hessian.
# Output
### KKT: KKT residual iteration sequence
### Portion: Portions of three cases among all iterations
### IdCon: indicator of whether convergence
### IdSing: indicator of singularity

include("sr1.jl")
function TRStoSQP(nlp, beta, zeta, ddelta, mu_1, rho, sigma, Max_Iter, EPS_Res, IdConst, IdB_k)

    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon
    IdCon, IdSing = 1, 0
    mu = mu_1
    # Initialize
    eps, k, X = 1, 0, [nlp.meta.x0]
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
    c_k, G_k = consjac(nlp, X[end])
    nabf_k = grad(nlp, X[end])
    P0 = Matrix(G_k*G_k')
    try
        invP = inv(Matrix(G_k*G_k'))
        P0 = invP
    catch
        IdSing = 1
    end
    if IdSing == 1
        return [], [], [], IdSing
    else
        lam_k = P0*(-G_k * nabf_k)
        nab_xL_k = nabf_k + G_k'lam_k
        KKT = [norm([nab_xL_k; c_k])]
        CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
        nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, lam_k)
        IdSetup = 1
    end

    # estimate Lipschitz constants Lipf and LipG
    eps_x = 0.1 * ones(nx)
    x_trial = X[end]+eps_x
    # Lipf, LipG:
    f_k = obj(nlp,X[end])
    f_trial = obj(nlp,x_trial)
    c_trial,G_trial = consjac(nlp,x_trial)
    LipG = norm(G_trial-G_k)/norm(eps_x)+1
    Lipf = norm(f_trial-f_k)/norm(eps_x)+1


        while KKT[end]>EPS_Res && k<Max_Iter
        ## Obtain est of gradient of Lagrange
            bnabf_k = rand(MvNormal(nabf_k, CovM))
            blam_k = P0 * (-G_k*bnabf_k)
            bnab_xL_k = bnabf_k + G_k' * blam_k
            push!(bnab_xL,bnab_xL_k)
            Delta = rand(Normal(0,(sigma)^(1/2)), nx, nx)
            bnab_x2L_k = nab_x2L_k + (Delta + Delta')/2
            push!(H,bnab_x2L_k)
            bKKT = norm([bnab_xL_k; c_k])
            beta_k = IdConst * beta + (1-IdConst) * (1/(k+1))^beta
            beta_max= IdConst * beta + (1-IdConst) * (1/(0+1))^beta
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

            ## Compute \bv_k:
            v_k = -G_k'* P0*c_k

            # compute control parameters
            eta1 = zeta * norm(v_k)/norm(c_k)
            tau_k = Lipf + LipG * mu + norm(B_k)
            alpha_k = beta_k/((4*eta1*tau_k+4*zeta)*beta_max)
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
        c_krs = c_k/norm(G_k)
        bnab_xL_krs  = bnab_xL_k/norm(B_k)
        bKKTrs = norm([bnab_xL_krs; c_krs])
        ## check_delta_k:
            ck_delta_k = norm(c_krs)/bKKTrs*delta_k
        ## tilde_delta_k:
            td_delta_k = norm(bnab_xL_krs)/bKKTrs*delta_k

        ## Select \gamma_k:
            gamma_ktrial = min(ck_delta_k/norm(v_k),1)

        phi_k = min(norm(B_k)/norm(G_k),1)

        lowerbd = 1/2*zeta*phi_k*alpha_k

        upperbd = lowerbd + ddelta*alpha_k^2

        if gamma_ktrial <= lowerbd
            gamma_k = lowerbd
        elseif gamma_ktrial >= upperbd
            gamma_k = upperbd
        else
            gamma_k = gamma_ktrial

        # Compute P_k:
            P_k = Diagonal(ones(nx))-G_k'* P0 * G_k

            # Compute Z_k:
            dense_G_k = Matrix(G_k')
            Q_k, = qr(dense_G_k)
            Z_k = Q_k[:,nlam+1:nx]
            nu=nx-nlam

        ## Compute \bu_k:
            m = Model(Ipopt.Optimizer);
            set_silent(m)
            @variable(m, u[1:nu]);
            @objective(m, Min, (bnabf_k+gamma_k*B_k*v_k)' *Z_k * u + 1/2 * u' * Z_k' * B_k * Z_k * u);
            @constraint(m,u'*u <= td_delta_k^2)
            optimize!(m);
            u_k=Float64[];
            for i=1:nu
              push!(u_k, callback_value(m,u[i]))
            end

        ## Compute the trial step:
            deltax_k = gamma_k*v_k + Z_k*u_k

        # compute predicted reduction:
            pred_k = bnabf_k'*deltax_k + 1/2*deltax_k'*B_k*deltax_k + mu*(norm(c_k + G_k*deltax_k)-norm(c_k))
            lower_bd_pred_k = -norm(bKKT)*delta_k + 1/2*norm(B_k)*delta_k^2
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
            k = k+1
            c_k, G_k = consjac(nlp, X[end])
            nabf_k = grad(nlp, X[end])
            P0= Matrix(G_k*G_k')
            try
                invP = inv(Matrix(G_k*G_k'))
                P0 = invP
            catch
                IdSing = 1
            end
            if IdSing == 1
                return [], [], [], IdSing
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
        return KKT[end], Por, 1, 0
    else
        return [],[], 0, 0
    end
end
