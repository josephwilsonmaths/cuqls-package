import LinearSampling.LossFunction

def Posterior(posterior_type, network, precision='single'):
    if posterior_type == 'NUQLS':
        posterior = LinearSampling.LossFunction.LeastSquaresLogitsPosterior('ntk', network, precision)
    elif posterior_type == 'NUQLSc':
        posterior = LinearSampling.LossFunction.CrossEntropyPosterior('ntk', network, precision)
    elif posterior_type == 'CUQLS':
        posterior = LinearSampling.LossFunction.LeastSquaresLogitsPosterior('ck', network, precision)    
    elif posterior_type == 'CUQLSc':
        posterior = LinearSampling.LossFunction.CrossEntropyPosterior('ck', network, precision)
    else:
        raise ValueError('Invalid posterior. Options are "NUQLS", "NUQLSc", "CUQLS", or "CUQLSc".')
    posterior.set_methodname(posterior_type)
    return posterior

            