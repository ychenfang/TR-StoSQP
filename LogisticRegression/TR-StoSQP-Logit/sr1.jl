function sr1(B,bnab_xL,X)
    yk = bnab_xL[end] - bnab_xL[end-1]
    sk = X[end] - X[end-1]
    Hk = B[end]
    quant1 = yk - Hk*sk
    quant2 = quant1'*sk
    if norm(yk) ==0 || norm(sk)==0
        return Hk
    elseif quant2 < 1e-8 * norm(sk) * norm(quant1)
        return Hk
    else
        H_new = Hk + (quant1*quant1')/quant2
        if norm(H_new)>=100
            H_new=B[1]      # if SR1 update is unstable, go back to Initialization
        end
        return H_new
    end
end
