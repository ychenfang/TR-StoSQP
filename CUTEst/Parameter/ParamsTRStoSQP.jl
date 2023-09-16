TRSto = Parameter.TRStoParams(true,
                    [0.5, 1],                   # const stepsize
                    [0.6, 0.8],                 # decay stepsize
                    10,                         # zeta
                    10,                         # delta
                    1,                          # mu_{-1}
                    1.5,                        # rho
                    100000,                     # Max_Iter
                    5,                          # Rep
                    1e-4,                       # EPS
                    [1e-8,1e-4,1e-2,1e-1])                        # Batchsize
