from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):


    loss = (np.sum(np.maximum((1 - (yTr * (w.T.dot(xTr)))), 0)) +
            (lambdaa * (w.T.dot(w))))

    gradient = ((-1 * (((np.maximum(1 - yTr * ((w.T.dot(xTr))), 0) > 0) * yTr).dot(xTr.T))).T +
                (2 * lambdaa * w))
    # YOUR CODE HERE

    return loss,gradient

