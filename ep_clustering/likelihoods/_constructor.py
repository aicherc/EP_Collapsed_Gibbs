import ep_clustering.likelihoods

def construct_likelihood(name, data, **kwargs):
    """ Construct likelihood function by name

    Args:
        name (string): name of likelihood. One of
            * "Gaussian"
            * "MixDiagGaussian"
            * "TimeSeries"
            * "AR"
            * "FixedVonMisesFisher"
        data (GibbsData): data
        **kwargs: arguments to pass to likelihood constructor

    Returns:
         likelihood (Likelihood): the appropriate likelihood subclass
    """
    if(name == "Gaussian"):
        return ep_clustering.likelihoods.GaussianLikelihood(data, **kwargs)
    elif(name == "MixDiagGaussian"):
        return ep_clustering.likelihoods.MixDiagGaussianLikelihood(data, **kwargs)
    elif(name == "TimeSeries"):
        return ep_clustering.likelihoods.TimeSeriesLikelihood(data, **kwargs)
    elif(name == "TimeSeriesRegression"):
        return ep_clustering.likelihoods.TimeSeriesRegressionLikelihood(data, **kwargs)
    elif(name == "AR"):
        return ep_clustering.likelihoods.ARLikelihood(data, **kwargs)
    elif(name == "StudentT"):
        return ep_clustering.likelihoods.StudentTLikelihood(data, **kwargs)
    elif(name == "FixedVonMisesFisher"):
        return ep_clustering.likelihoods.FixedVonMisesFisherLikelihood(data, **kwargs)
    elif(name == "VonMisesFisher"):
        return ep_clustering.likelihoods.VonMisesFisherLikelihood(data, **kwargs)
    else:
        raise ValueError("Unrecognized name {0}".format(name))
    return


