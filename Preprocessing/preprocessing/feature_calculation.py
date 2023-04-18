import numpy as np

def convolve_filterbank(x,t=None,L=1,skip=1,filter=None,filtershape='square', mode='valid',**kwargs):
    '''
    Input

        skip: int, how many convolution to skip
        L: window length of filters
        filtershape: tri, square, or both.
            tri produces filter with triangualr pattern of length 2*L-1 e.g. L=2 (0,0.5,1,0.5,1)
            square e.g. L=2 (0,1,1,1,0)
            both applies tri then square
    '''
    if filter is None:

        if filtershape == 'tri' or filtershape == 'both':
            filter = np.concatenate([np.arange(L)+1,np.arange(L)[::-1][:-1]])
            # Normalize so the filter sums up to 1
        elif filtershape == 'square':
            filter = np.ones(L*2-1)

        filter = filter / filter.sum()

    x = np.convolve(x,filter,mode=mode,**kwargs)

    if t is not None and L > 1:
        t = t[(L-1):-(L-1)]

    if filtershape == 'both':
        x,t = convolve_filterbank(x,t,L,skip=1,filtershape='square',mode=mode,**kwargs)

    ## Skip
    x = x[::skip]
    t = t[::skip]

    return x,t

def threshold_angle(angle,dist,cutoff=0.2,n=1):

    idx_discard = dist < cutoff
    idx_discard_ = idx_discard

    for ii in range(n-1):
        idx_discard_[n:]  = idx_discard_[n:]  + idx_discard[:-n]
        idx_discard_[:-n] = idx_discard_[:-n] + idx_discard[n:]

    angle[idx_discard] = 0

    return angle

def calculate_distance(t,X,Y,T=1,K=1,L=1,**kwargs):
    """
    Calculate distance traveled between each time points
    T = How many time points to look back
    K = How many time points to skip
    L = window length of filters
    """
    t = np.array(t)
    coords = np.array([X,Y]).transpose()
    dist = np.linalg.norm(coords[T:] - coords[:-T],axis=1)
    dist = dist[::K]
    t = t[T:][::K]
    dist,t = convolve_filterbank(dist,t,L,**kwargs)
    return dist, t


def calculate_linear_speed(t,X,Y,T=1,K=1,L=1,**kwargs):
    """
    Calculate speed fish traveled between each time points
    T = How many time points to look back
    K = How many time points to skip
    """
    t = np.array(t)
    dist,t_dist = calculate_distance(t,X,Y,T,K=1)
    dt = t[T:] - t[:-T]
    speed = dist/dt
    speed = speed[::K]
    t = t[T:][::K]
    speed,t = convolve_filterbank(speed,t,L,**kwargs)
    return speed, t


def calculate_acceleration(t,X,Y,T=1,K=1,L=1,**kwargs):
    """
    Calculate acceleration of fish traveled between each time points
    T = How many time points to look back
    K = How many time points to skip
    """
    t = np.array(t)
    v, t_v = calculate_linear_speed(t,X,Y,T,K=1)
    dv = v[T:] - v[:-T]
    dt = t_v[T:] - t_v[:-T] # time from x[i-T] to x[i]
    a = dv/dt
    a = a[::K]
    t = t_v[T:][::K]
    a,t = convolve_filterbank(a,t,L,**kwargs)
    return a,t


def calculate_angle(t,X,Y,d=None,cutoff=0,T=1,K=1,L=1,n=2,replace_NaNs=True,**kwargs):
    """
    Calculate angle between each time points
    T = How many time points to look back
    K = How many time points to skip
    0 vectors results in 0 angle
    """
    t = np.array(t)
    coords = np.array([X,Y]).transpose()
    v1 = coords[T:-T] - coords[:-2*T]
    v2 = coords[2*T:] - coords[T:-T]
    t_vector = t[T:-T]
    dt = t[2*T:]- t[:-2*T]
    x = np.sum(v1*v2,axis=1) # dot product
    y = v2[:,1]*v1[:,0] - v2[:,0]*v1[:,1] #
    angle = np.arctan2(y,x)
    ##
    if replace_NaNs:
        angle[np.isnan(angle)] = 0
    ##
    angle = angle[::K]
    ## Threshold angle based on distance
    if d is not None:
        angle = threshold_angle(angle,d[1:],cutoff,n)
    ## Index time/dt
    t_ = t_vector[::K]
    dt = dt[::K]

    angle,t_ = convolve_filterbank(angle,t_,L,**kwargs)

    return angle, t_, dt


def calculate_angular_speed(t,X,Y,d=None,cutoff=0,T=1,K=1,L=1,**kwargs):
    """
    Calculate speed fish traveled between each time points
    T = How many time points to look back
    K = How many time points to skip
    """
    t = np.array(t)
    angle, t_angle, dt = calculate_angle(t,X,Y,T=T,d=d,cutoff=cutoff,K=1,L=1)
    t_vector = t_angle
    speed = angle/dt
    speed = speed[::K]
    t_ = t_vector[::K]
    speed,t_ = convolve_filterbank(speed,t_,L,**kwargs)
    return speed, t_


def calculate_angular_acceleration(t,X,Y,d=None,cutoff=0,T=1,K=1,L=1,**kwargs):
    """
    Calculate acceleration of fish traveled between each time points
    T = How many time points to look back
    K = How many time points to skip
    """
    t = np.array(t)
    v,t_v = calculate_angular_speed(t,X,Y,T=T,d=d,cutoff=cutoff,K=1,L=1)
    dv = v[T:] - v[:-T]
    dt = (t_v[T:] - t_v[:-T])
    a = dv/dt
    a = a[::K]
    t = t_v[T:][::K]
    a,t = convolve_filterbank(a,t,L,**kwargs)
    return a,t


def calculate_distance_from_center(X,Y,t,L=1,**kwargs):
    '''
    Calculate distance from center where center is estimated as halfway point between min and max of X and Y.
    '''
    X0 = (np.max(X) + np.min(X))/2
    Y0 = (np.max(Y) + np.min(Y))/2
    dist = np.sqrt(np.square(X-X0) + np.square(Y-Y0))
    dist,t = convolve_filterbank(dist,t,L,**kwargs)
    return dist,t
