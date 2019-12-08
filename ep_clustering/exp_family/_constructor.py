import ep_clustering.exp_family

def construct_exponential_family(name, num_dim, **kwargs):
    """ Construct exponential family site approximation by name

    Args:
        name (string): name of exponential family:
            * NormalFamily
            * DiagNormalFamily
        num_dim (int): dimension of exponential family
        **kwargs: arguments to pass to the exponential family constructor

    Returns:
        expfam (ExponentialFamily): the appropriate exponential family subclass
    """
    if(name == "NormalFamily"):
        return exp_family.NormalFamily(num_dim=num_dim, **kwargs)
    elif(name == "DiagNormalFamily"):
        return exp_family.DiagNormalFamily(num_dim=num_dim, **kwargs)
    else:
        raise ValueError("Unrecognized name {0}".format(name))
    return
