from scipy.fft import fft, fftfreq
import pywt

def apply_fft(df=None, t=None, y=None):

    if df is not None:
        t = df.abstime.values
        y = df.Value.values
    else:
        if t is None or y is None:
            raise Exception('Please specify either df or t and y')

    yf = fft(y)
    N = len(t)
    T = (t.max() - t.min())/(N-1)
    tf = fftfreq(N, T)[:N//2]
    yf = 2.0/N * np.abs(yf[:N//2])
    df = pd.DataFrame({'Freq':tf,'Sp':yf})

    return df


def apply_cwt(scales, wavelet, t=None,y=None,df=None,**kwargs):
    """
    Input
        t: vector of time, the increment must be constant
        y: vector of signal
        scales: vector of scales for cwt
        wavelet: type of wavelet
    return
        data frame of cross correlation with time as columns and frequency as index
    """
    if df is not None:
        t = df.abstime.values
        y = df.Value.values
    else:
        if t is None or y is None:
            raise Exception('t and y should be provided')
    ## Calculate dt, time interval
    dt = t[1:] - t[:-1]
    if np.all(dt == dt[0]):
        dt = dt[0]
    else:
        raise Exception('t should be spaced at constant interval')

    [coefficients, frequencies] = pywt.cwt(y, scales, wavelet, dt, **kwargs)

    df_wt = pd.DataFrame(coefficients,index=frequencies,columns=t)

    return df_wt
