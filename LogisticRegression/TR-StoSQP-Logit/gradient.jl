function grad_logit(X,y,x)
    batch_size = size(X,1)
    nx = size(X,2)
    gradlogit = zeros(nx)
    for i=1:batch_size
        yi = y[i]
        Xi = X[i,:]
        quant1 = -yi*Xi+1/(1+exp(-yi*(Xi'*x)))*yi*Xi
        gradlogit = gradlogit+quant1
    end
    return gradlogit/batch_size
end
