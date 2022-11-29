function obj_logit(X,y,x)
    batch_size = size(X,1)
    nx = size(X,2)
    objlogit = 0
    for i=1:batch_size
        yi = y[i]
        Xi = X[i,:]
        quant1 = log(1+exp(-yi*(Xi'*x)))
        objlogit = objlogit+quant1
    end
    return objlogit/batch_size
end
