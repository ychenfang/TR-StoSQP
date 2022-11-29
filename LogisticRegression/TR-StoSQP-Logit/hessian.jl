function hess_logit(X,y,x)
    batch_size = size(X,1)
    nx = size(X,2)
    hesslogit = zeros(nx,nx)
    for i=1:batch_size
        yi = y[i]
        Xi = X[i,:]
        quant1 = exp(-yi*(Xi'*x))/(1+exp(-yi*(Xi'*x)))^2*(yi*Xi)*(yi*Xi)'
        hesslogit = hesslogit+quant1
    end
    return hesslogit/batch_size
end
