from numpy.fft import ifft

def calc_translocation(F1,F2):
    """
    Find the optimal translocation phase to minimize convolution through cross-power spectrum calculation
    """
    cp_spectrum = F1*np.conjugate(F2)/np.abs(F1*np.conjugate(F2))
    conv = ifft(cp_spectrum)

from scipy import signal

test_name = 'sin5'
t = np.arange(10000)/10000*2*np.pi*5
T = (t.max() - t.min())/(N-1)
x1 = np.sin(t)
x2 = np.sin(t + np.pi)
window = np.hamming(len(t))
F1 = fft(x1)
F2 = fft(x2)

cp_spectrum = F1*np.conjugate(F2)/np.abs(F1*np.conjugate(F2))
conv = ifft(cp_spectrum)

F1*np.conjugate(F2)
np.abs(F1*np.conjugate(F2))

## Plot frequency domain
N = len(t)
T = (t.max() - t.min())/(N-1)
tf = fftfreq(N, T)[:N//2]
yf1 = 2.0/N * np.abs(F1[:N//2])
yf2 = 2.0/N * np.abs(F2[:N//2])
df_plot = pd.DataFrame({'Frequency':tf, 'F1':yf,'F2':yf2})
df_plot = df_plot.melt(id_vars=['Frequency'],var_name='Fun',value_name='y')
ax = sns.lineplot(data=df_plot, x='Frequency', y = 'y', hue = 'Fun')
plt.xlim(0,1)
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_Freq.svg'.format(test_name)))
plt.close()

## Plot the iFFT
x1i = ifft(F1).real
x2i = ifft(F2).real
df_plot = pd.DataFrame({'Time':t, 'X1':x1i, 'X2':x2i})
df_plot = df_plot.melt(id_vars=['Time'],var_name='Fun',value_name='y')
ax = sns.lineplot(data=df_plot, x='Time', y = 'y', hue = 'Fun')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_recon.svg'.format(test_name)))
plt.close()

## Plot convolutions
df_plot = pd.DataFrame({'Time':t, 'X1':x1, 'X2':x2, 'Cross-spectrum': conv.real})
df_plot = df_plot.melt(id_vars=['Time','Cross-spectrum'],var_name='Fun',value_name='y')
ax = sns.lineplot(data=df_plot, x='Time', y = 'y', hue = 'Fun')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}.svg'.format(test_name)))
plt.close()

ax = sns.lineplot(data=df_plot, x='Time', y = 'Cross-spectrum')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_conv.svg'.format(test_name)))
plt.close()

df_plot = pd.DataFrame({'Time':t, 'Cross-spectrum': conv.real})
ax = sns.lineplot(data=df_plot, x='Time', y = 'Cross-spectrum')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_conv1.svg'.format(test_name)))
plt.close()

conv2 = np.correlate(x1,x2,'full')
t2 = np.concatenate([-t[1:][::-1],t])
df_plot = pd.DataFrame({'Time':t2, 'Cross-spectrum': conv2})
ax = sns.lineplot(data=df_plot, x='Time', y = 'Cross-spectrum')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_conv2.svg'.format(test_name)))
plt.close()

w = np.concatenate([np.arange(len(t))+1,np.arange(len(t))[::-1][:-1]])
conv3 = conv2/w
df_plot = pd.DataFrame({'Time':t2, 'Cross-spectrum': conv3})
ax = sns.lineplot(data=df_plot, x='Time', y = 'Cross-spectrum')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_conv3.svg'.format(test_name)))
plt.close()

conv4 = np.correlate(x1*window,x2*window,'full')
t2 = np.concatenate([-t[1:][::-1],t])
df_plot = pd.DataFrame({'Time':t2, 'Cross-spectrum': conv2})

ax = sns.lineplot(data=df_plot, x='Time', y = 'Cross-spectrum')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_conv4.svg'.format(test_name)))
plt.close()

w = np.concatenate([np.arange(len(t))+1,np.arange(len(t))[::-1][:-1]])
conv5 = conv4/w
df_plot = pd.DataFrame({'Time':t2, 'Cross-spectrum': conv3})
ax = sns.lineplot(data=df_plot, x='Time', y = 'Cross-spectrum')
plt.savefig(os.path.join(logdir_plots,'Translocation_test-{}_conv5.svg'.format(test_name)))
plt.close()


#### Wavelet transform
import pywt
from scipy.fft import fft, fftfreq

wave_types = ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']

test_name = 'sin5'
n = 500
t = np.arange(n)/n*2*np.pi*3
x1 = np.sin(t**2)

# cA, cD = pywt.dwt([1, 2, 3, 4], 'db1')
# cA, cD = pywt.dwt(x1, 'db1')
# tw = t[::2]
# df_plot = pd.DataFrame({'Time':tw, 'cA': cA})
# ax = sns.lineplot(data=df_plot, x='Time', y = 'cA')
# plt.savefig(os.path.join(logdir_plots,'Wavelet_cA_{}.svg'.format(test_name)))
# plt.close()
# df_plot = pd.DataFrame({'Time':tw, 'cD': cD})
# ax = sns.lineplot(data=df_plot, x='Time', y = 'cD')
# plt.savefig(os.path.join(logdir_plots,'Wavelet_cD_{}.svg'.format(test_name)))
# plt.close()

#### Wavelet cwt
logdir_plots = os.path.join('figures')
os.makedirs(logdir_plots,exist_ok=True)

wavlist = pywt.wavelist(kind='continuous')

test_name = 'chirp'
time = t
dt = time[1] - time[0]
signal = x1
scales = np.arange(1,12880,10)

for waveletname in ['mexh',"gaus1","gaus4","gaus8",'cgau1','cgau4','cgau8','morl']:
    # waveletname = 'db10'
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = coefficients
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    cmap = plt.cm.seismic
    #
    fig, ax = plt.subplots()
    im = ax.contourf(time, frequencies, power, contourlevels, extend='both',cmap=cmap,center=0)
    #
    ax.set_title(waveletname, fontsize=20)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    #
    # cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, orientation="vertical")
    #
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_plots,'Wavelet_cwt_{}_{}.svg'.format(test_name,waveletname)))
    plt.close()


df_plot = pd.DataFrame({'Time':t, 'x': x1})
ax = sns.lineplot(data=df_plot, x='Time', y = 'x')
plt.savefig(os.path.join(logdir_plots,'Wavelet_cwt_{}_x.svg'.format(test_name)))
plt.close()
#

#### FFT
F1 = fft(x1)
T = (t.max() - t.min())/(N-1)
N = len(F1)
T = (t.max() - t.min())/(N-1)
tf = fftfreq(N, T)[:N//2]
yf1 = 2.0/N * np.abs(F1[:N//2])
#
df_plot = pd.DataFrame({'Freq':tf, 'Sp': yf1})
ax = sns.lineplot(data=df_plot, x='Freq', y = 'Sp')
plt.savefig(os.path.join(logdir_plots,'Wavelet_fft_{}_x.svg'.format(test_name)))
plt.close()

#### DWT test
data = x1
waveletname = 'sym5'

fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
coeffs = []
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, waveletname)
    coeff_d.shape
    coeffs.append(np.repeat(coeff_d,2**ii))
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])

plt.tight_layout()
plt.savefig(os.path.join(logdir_plots,'Wavelet_dwt_{}.svg'.format(test_name)))
plt.close()

n_min = min([len(coeff) for coeff in coeffs])
coeffs_array = np.array([coeff[(np.linspace(1,len(coeff),n_min)).astype(int)-1] for coeff in coeffs])
cmap = plt.cm.seismic
ax=sns.heatmap(coeffs_array,cmap=cmap,center=0)
ax.set_ylabel('Level', fontsize=18)
ax.set_xlabel('Time', fontsize=18)

plt.tight_layout()
plt.savefig(os.path.join(logdir_plots,'Wavelet_dwt_array_{}.svg'.format(test_name)))
plt.close()


#### Create mixed signals
test_name = 'mixed_signal'
t = np.linspace(1,60,5000)
f2 = 1
main_wave= np.cos(t*2*np.pi*(f2))

for waveletname in ['mexh',"gaus1","gaus4","gaus8",'cgau1','cgau4','cgau8','morl']:
    for percent_occurence in [0.2,0.1,0.05,0.01]:
        logdir_ = os.path.join(logdir_plots,
                               'percent_occurence_test',
                               waveletname,
                               'percent_occurence={}'.format(percent_occurence))
        os.makedirs(logdir_,exist_ok=True)
        ## Create artificial signal
        f2 = 10
        num_occurence = 5
        cycle_length = np.int(len(t)/num_occurence)
        duration = int(cycle_length * percent_occurence)
        secondary_wave = np.cos(t[:duration]*2*np.pi*(f2))
        secondary_wave = np.concatenate([secondary_wave,np.zeros(cycle_length-duration)])
        secondary_wave = np.tile(secondary_wave,num_occurence)
        if len(t)-len(secondary_wave) !=0 :
            secondary_wave = np.concatenate([secondary_wave,np.zeros(len(t)-len(secondary_wave))])
        combined_wave = main_wave + secondary_wave
        #
        fig, ax = plt.subplots()
        ax.plot(t,main_wave,label='consistent signal')
        ax.plot(t,secondary_wave,label='sparse signal')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Signal')
        ax.legend()
        plt.savefig(os.path.join(logdir_,'Wavelet_{}_{}.svg'.format(test_name,'signal')))
        plt.close()
        #
        df_fft = apply_fft(t=t,y=combined_wave)
        sns.lineplot(data=df_fft,x='Freq',y='Sp')
        plt.savefig(os.path.join(logdir_,'Wavelet_{}_{}.svg'.format(test_name,'FFT')))
        plt.close()
        #
        time   = t
        signal = combined_wave
        scales = (np.arange(128)+1)
        # scales = 2**np.arange(10)
        dt=t[1]-t[0]
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = coefficients
        power = (abs(coefficients)) ** 2
        period = 1. / frequencies
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)
        cmap = plt.cm.seismic
        #
        fig, ax = plt.subplots()
        im = ax.contourf(time, frequencies, power, contourlevels, extend='both',cmap=cmap,center=0)
        #
        ax.set_title(waveletname, fontsize=20)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_xlabel('Time', fontsize=18)
        #
        # cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
        fig.colorbar(im, orientation="vertical")
        #
        plt.tight_layout()
        plt.savefig(os.path.join(logdir_,'Wavelet_{}_{}.svg'.format(test_name,'cwt')))
        plt.close()
