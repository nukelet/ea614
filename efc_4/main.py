import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from kaiser import kaiser

def spectrum(y, filename="spectrum.png"):
    Y = np.abs(np.fft.fft(y))
    w = np.linspace(0, 2*np.pi, Y.size)
    plt.figure()
    plt.plot(w, Y/np.max(Y))
    plt.xlabel("$\Omega$ [rad]")
    plt.ylabel('|$Y(e^{j\Omega})|$')
    plt.grid(True)
    plt.xlim((0, 2*np.pi))
    plt.savefig(f"latex/images/{filename}")
    # plt.show()
    return Y, w

def plot_kaiser_response(freqs,
    filename="kaiser_freq_response.png"):
    plt.figure()
    colors = ["blue", "green", "red", "orange", "yellow"]
    legend = []
    for i, (pass_freq, cutoff_freq) in enumerate(freqs):
        Y = np.abs(np.fft.fft(kaiser(pass_freq, cutoff_freq)))
        w = np.linspace(0, 2*np.pi, Y.size)
        plt.plot(w, Y/np.max(Y), color=colors[i])
        plt.xlabel("$\Omega$ [rad]")
        plt.ylabel('|$Y(e^{j\Omega})|$')
        plt.grid(True)
        plt.xlim((0, 2*np.pi))
        legend.append(f"$\Omega_p = {pass_freq}$, $\Omega_r = {cutoff_freq}$")
    plt.legend(legend)
    plt.savefig(f"latex/images/{filename}")
    # plt.show()

# reduce the sample rate by a factor of `m`
def downsample(data, m):
    downsampled = []
    # pick every m-th samples
    for i in range(0, data.size, m):
        downsampled.append(data[i])
    downsampled = np.array(downsampled)
    return downsampled

def filter_signal(y):
    return np.convolve(y, kaiser(0.45, 0.5))

def creed_overcome():
    fs, y = sio.wavfile.read("data/creed_overcome.wav")

    # convert from stereo to mono (2 channels to 1 channel)
    y = y[:,0] + y[:,1] 
    spectrum(y, "spectrum.png")

    m = 10
    y_downsampled = downsample(y, m)
    spectrum(y_downsampled, "downsampled_spectrum.png")
    # sd.play(y_downsampled, fs/m, blocking=True)
    plot_kaiser_response(
        [(0.45, 2), (0.45, 0.5), (1.5, 2)])

    y_filtered = filter_signal(y)
    spectrum(y_filtered, "filtered_spectrum.png")

    y_filtered_downsampled = downsample(y_filtered, 6)
    spectrum(y_filtered_downsampled,
        "filtered_downsampled_spectrum.png")
    
    sd.play(y_filtered_downsampled, fs/6, blocking=True)
    # sd.play(y_filtered, fs, blocking=True)


def piano_note():
    fs, y = sio.wavfile.read("data/piano_note.wav")
    y = y[:,0] + y[:,1]
    print(f"audio sample rate: {fs}, samples: {y.size}")
    spectrum(y, "piano_note_spectrum.png")
    Y = np.abs(np.fft.fft(y))
    k_max = np.argmax(Y)
    freq_max = k_max * fs / Y.size
    print(f"max index: {k_max}, freq: {freq_max}")
    ind = np.argpartition(Y, -10)[-10:]
    print(list(zip(ind, ind * fs / Y.size, Y[ind])))

def eeg_spectrum_analog(data):
    fs = 250
    Y = np.abs(np.fft.fft(data))
    size = int(Y.size/2)
    k = Y[:size].argmax()
    print(f"max: {(k, k * fs / Y.size)}")
    f_bins = np.linspace(0, fs/2, size)
    plt.figure()
    plt.plot(f_bins, Y[0:size]/Y.max())
    plt.savefig("latex/images/eeg_spectrum.png")
    

def eeg():
    data = sio.loadmat("data/EEG.mat")["EEG"].flatten()
    print(data)
    eeg_spectrum_analog(data)


def main():
    # creed_overcome()
    # piano_note()
    eeg()

if __name__ == "__main__":
    main()