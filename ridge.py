import numpy as np


def ridge(w, xTr, yTr, lambdaa):
    loss = ((w.T.dot(xTr) - yTr).dot((w.T.dot(xTr) - yTr).T) +
            (w.T.dot(w) * lambdaa))

    gradient = (((xTr.dot(xTr.T)) * 2).dot(w) -
                (xTr.dot(yTr.T) * 2) + (lambdaa * w * 2))

    return loss, gradient
