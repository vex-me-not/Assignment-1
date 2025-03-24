
import numpy as np
import matplotlib.pyplot as plt


def encode_sex(sex):
    """
    Method used to encode the entries of the column sex
    Male --> 0
    Female --> 1
    Other --> 2
    """

    if sex=="Male":
        return 0
    elif sex=="Female":
        return 1
    else:
        return 2