import numpy as np


def grdescent(func, w0, stepsize, maxiter, tolerance=1e-02):
    w = w0
    l_init, g = func(w)  # return loss and gradient from loss function

    for i in range(maxiter):
        # Make sure to include the tolerance variable to stop early if the norm of the gradient is
        # less than the tolerance value (you can use the function norm(x))
        if np.linalg.norm(g) < tolerance:
            break

        w = w - stepsize * g  # update w

        l_new, g = func(w)
        # I increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
        # and decrease it by a factor 0.5 if the loss went up
        if l_new >= l_init:
            stepsize = stepsize * 0.5
        else:
            stepsize = stepsize * 1.01

        l_init = l_new  # update loss for each iteration

        if stepsize < 2.2204e-14:
            break

    return w
