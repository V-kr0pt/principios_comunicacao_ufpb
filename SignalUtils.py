import numpy as np


class SignalUtils:
    def __init__(self, signal, fs, N):
        self.self.signal = signal
        self.self.fs = fs
        self.self.N = N
        
    def calculate_psd(self):
        freqs = np.fft.fftfreq(len(self.signal), 1/self.fs)
        fft_signal = np.fft.fft(self.signal) / self.N
        psd = np.abs(fft_signal) ** 2 
        return freqs, psd

    def determine_bandwidth(self, power_threshold=0.95, verbose=True):
        freqs, psd = self.calculate_psd(self.signal, self.fs, self.N)

        # Now we only use the half of the frequencies
        freqs = freqs[:len(freqs)//2]
        psd = 2*psd[:len(psd)//2]    

        # Find the maximum frequency
        max_freq_index = np.argmax(psd)
        max_freq = freqs[max_freq_index]    

        # Calculate the total power
        total_power = np.sum(psd)

        # Initialize variables for cumulative power and bandwidth calculation
        cumulative_power = 0
        lower_index = max_freq_index
        upper_index = max_freq_index

        # Expand around the peak frequency to find the bandwidth (try to optimize this after)
        while cumulative_power / total_power <= power_threshold:
            if lower_index > 0:
                lower_index -= 1
            if upper_index < len(freqs) - 1:
                upper_index += 1

            cumulative_power = np.sum(psd[lower_index:upper_index+1])

        # Find the frequency where the cumulative power exceeds the threshold
        bandwidth = (freqs[lower_index], freqs[upper_index])

        if verbose:
            print(f"Bandwidth: {bandwidth[0]} Hz - {bandwidth[1]} Hz")
            print(f"Max amplitude frequency: {max_freq} Hz")
            print(f"Frequency range: {np.min(freqs)} Hz - {np.max(freqs)} Hz")
            print(f"self.signal total power: {total_power}")

        return bandwidth

if __name__ == '__main__':

    # creating time and frequency vectors
    fs = 10000 # sampling frequency (Bandwidth = 2000Hz)
    Ts = 1/fs # sampling period or time step: t[1] - t[0] dt 
    T_total = 2 # total time of self.signal (s)
    N = int(T_total/Ts) # number of samples

    t = np.linspace(0, T_total, N) # time vector (s)
    #frequencies = np.fft.fftfreq(t.size, Ts) # frequency vector (Hz)

    signal_a = signal_a = np.exp(-10*t)

    signal_utils = SignalUtils(signal_a, fs, N)
    bandwith = signal_utils.determine_bandwidth(power_threshold=0.95, verbose=True)