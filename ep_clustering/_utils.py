"""

Utilities functions

"""
import joblib
import os
import errno
import numpy as np
import pandas as pd
import logging
import types
from copy import deepcopy

logger = logging.getLogger(name=__name__)


# Handling Doc-string inheritance
def fix_docs(cls):
    """
    This will copy missing documentation from parent classes.

    Arguments:
        cls (class): class to fix up.

    Returns:
	cls (class): the fixed class.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, property) and not func.fget.__doc__:
            for parent in cls.__bases__:
                parprop = getattr(parent, name, None)
                if parprop and getattr(parprop.fget, '__doc__', None):
                    newprop = property(fget=func.fget,
                                       fset=func.fset,
                                       fdel=func.fdel,
                                       __doc__=parprop.fget.__doc__)
                    setattr(cls, name, newprop)
                    break

    return cls

# Custom Dictionary Class with additional functionality
@fix_docs
class Map(dict):
    """
    Customized dict class
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return Map(super(Map, self).copy())

    def deepcopy(self):
        return Map(deepcopy(super(Map,self).copy()))

    def save(self, filename, filetype = None, compress=0):
        """ Save dictionary to file """
        if filetype is None:
            filetype = filename.split(".")[-1]

        if filetype == "p":
            joblib.dump(self.__dict__, filename, compress=compress)
        else:
            raise ValueError("Unrecognized filetype '{0}'".format(filetype))
        return

    def load(self, filename, filetype = None):
        """ Load dictionary from file """
        if filetype is None:
            filetype = filename.split(".")[-1]

        if filetype == "p":
            obj = joblib.load(filename)
        else:
            raise ValueError("Unrecognized filetype '{0}'".format(filetype))

        if isinstance(obj, dict):
            for key, value in obj.items():
                super(Map, self).__setitem__(key, value)
                self.__dict__.update({key: value})
        else:
            raise ValueError(
                    "Unrecognized obj type '{0}'".format(type(obj)))
        return self

def getFromNestDict(dataDict, mapList):
    for k in mapList:
        dataDict = dataDict[k]
    return dataDict

def setFromNestDict(dataDict, mapList, value):
    getFromNestDict(dataDict, mapList[:-1])[mapList[-1]] = value
    return

def logsumexp(x, weights=None):
    """ LogSumExp trick with optional weights

    Args:
        x (ndarray): input
        weights (ndarray): weight

    Returns:
        out (double): log(sum(weights * exp(x)))
    """
    if weights is None:
        weights = np.ones_like(x)
    a = np.max(x)
    out = np.log(np.sum(weights * np.exp(x-a))) + a
    return out

# This should really be in a dataset 'class'
def convert_df_to_matrix(df, value_name = "y",
        row_index="observation",
        col_index="dimension"):
    """ Converts df to y matrix and y_counts matrix

    Args:
        df (pd.DataFrame): raw data with
            index (row_index, col_index) and column value_name
        value_name (string): name column in df for output mean and count
        row_index (int): name of df index for rows for output
        col_index (int): name of df index for columns of output

    Returns:
        mean_matrix (np.ndarray): row by column mean values (nan for missing)
        count_matrix (np.ndarray): row by column count of values

    """
    if value_name not in df.columns:
        raise ValueError("value_name {0} not in df".format(value_name))
    if row_index not in df.index.names:
        raise ValueError("row_index {0} not in df index".format(row_index))
    if col_index not in df.index.names:
        raise ValueError("col_index {0} not in df index".format(col_index))
    agg_df = df.groupby(level=[row_index, col_index])[value_name].aggregate(
            mean=np.mean, count=np.size,
            )

    mean_matrix = agg_df['mean'].reset_index().pivot(
            index = row_index,
            columns = col_index,
            values = 'mean').values

    count_matrix = agg_df['count'].reset_index().pivot(
            index = row_index,
            columns = col_index,
            values = 'count').values
    count_matrix = np.nan_to_num(count_matrix)

    return mean_matrix, count_matrix


def convert_matrix_to_df(y, observation_name = "y"):
    """ Converts y matrix to df format

    Args:
        y (np.ndarray): observation by dimension matrix of values

    """
    n_rows, n_cols = np.shape(y)
    row_index = pd.Index(np.arange(n_rows), name = 'observation')
    col_index = pd.Index(np.arange(n_cols), name = 'dimension')

    out = pd.DataFrame(y, index=row_index, columns=col_index).stack()
    out_df = pd.DataFrame(out)
    out_df.columns = [observation_name]

    return out_df

def make_path(path):
    """ Helper function for making directories """
    if path is not None:
        if not os.path.isdir(path):
            if os.path.exists(path):
                raise ValueError(
        "path {0} is any existing file location!".format(path)
                            )
            else:
                try:
                    os.makedirs(path)
                except OSError as e:
                    logger.warning(e.args)
                    if e.errno != errno.EEXIST:
                        raise e
    return




