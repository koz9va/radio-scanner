import osmosdr
import structlog
import numpy as np
from gnuradio import gr, blocks, analog
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as sci_find_peaks, ShortTimeFFT
from scipy.signal.windows import nuttall

structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()


def median_truncate(data: np.ndarray, div_factor: int) -> np.ndarray:
    out = np.zeros(len(data) // div_factor, dtype=data.dtype)
    sg_size = int(len(data) / div_factor)  # segment size
    for i in range(len(out) - 1):
        out[i:i + sg_size] = np.median(data[i:i + sg_size])

    return out

def shrink_array_by_median(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Shrinks a 1D NumPy array by a factor, taking the median of each segment.

    Parameters:
      arr: 1D numpy array
      factor: integer factor by which to shrink the array

    Returns:
      A new 1D numpy array where each element is the median of a segment of length 'factor'
    """
    n = len(arr)
    # Trim array so length is a multiple of factor
    trimmed_length = (n // factor) * factor
    trimmed = arr[:trimmed_length]
    # Reshape into a 2D array with shape (-1, factor)
    reshaped = trimmed.reshape(-1, factor)
    # Take the median along axis 1
    return np.median(reshaped, axis=1)


class HackRfScanner(gr.top_block):
    source: osmosdr.source
    agc: analog.agc_cc
    head: blocks.head
    sink: blocks.vector_sink_c

    center_freq: float
    sample_rate: float
    num_samples: int

    def set_center_freq(self, center_freq):
        self.source.set_center_freq(center_freq)
        self.center_freq = center_freq

    def __init__(self, center_freq: float, sample_rate: float, num_samples: int):
        gr.top_block.__init__(self)
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        self.source = osmosdr.source(args='hackrf')
        self.source.set_center_freq(self.center_freq)
        self.source.set_sample_rate(self.sample_rate)
        self.source.set_gain(20)

        self.agc = analog.agc_cc(1e-3, 1.0, 1.0)
        self.agc.set_max_gain(10000)


        # taps = filter.firdes.low_pass(
        #     1.0,
        #     self.sample_rate,
        #     250e3,
        #     50e3,
        #     filter.window.WIN_HAMMING,
        #     6.76
        # )

        # self.lpf = filter.fir_filter_ccf(1, taps)

        self.head = blocks.head(gr.sizeof_gr_complex, self.num_samples)
        self.sink = blocks.vector_sink_c()

        self.connect(self.source, self.agc, self.head, self.sink)

    def get_samples(self):
        return np.array(self.sink.data(), dtype=np.complex64)


def scan_frequency_range(start_freq: float,
                         stop_freq: float,
                         step: float,
                         sample_rate: float,
                         num_samples: int,
                         min_bandwidth: float,
                         max_bandwidth: float,
                         noise_above: float = 0.05,
                         visualize: bool = True) -> list[float]:
    # detected_transmissions = []
    all_freqs_list = []
    all_power_list = []
    i = 0
    for freq in np.arange(start_freq, stop_freq, step):
        scanner = HackRfScanner(float(start_freq), sample_rate, num_samples)
        scanner.start()
        scanner.set_center_freq(freq)
        scanner.wait()

        samples = scanner.get_samples()

        np.save(f'samples/samples_{i}.npy', samples, allow_pickle=True)
        # samples = np.load(f'samples/samples_{i}.npy')
        i += 1

        scanner.stop()

        fft_data = np.fft.fftshift(np.fft.fft(samples))
        power_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
        fft_bins = np.fft.fftshift(np.fft.fftfreq(len(samples), 1.0/sample_rate))

        abs_freqs = freq + fft_bins

        all_freqs_list.append(abs_freqs)
        all_power_list.append(power_db)


    all_freqs = np.concatenate(all_freqs_list)
    all_power = np.concatenate(all_power_list)
    sort_idx = np.argsort(all_freqs)
    all_freqs_sorted = all_freqs[sort_idx]
    all_power_sorted = all_power[sort_idx]

    noise_median = float(np.median(all_power))

    del all_freqs
    del all_power

    # windowed_freqs = shrink_array_by_median(all_freqs_sorted, 100)
    # windowed_power = shrink_array_by_median(all_power_sorted, 100)
    windowed_freqs = all_freqs_sorted.reshape(-1, 1000).mean(axis=1)
    windowed_power = all_power_sorted.reshape(-1, 1000).mean(axis=1)
    # windowed_freqs = savgol_filter(all_freqs_sorted, 101, 2)
    # windowed_power = savgol_filter(all_power_sorted, 101, 2)

    delta_f = sample_rate / len(windowed_freqs)
    min_width_bins = min_bandwidth / delta_f
    max_width_bins = max_bandwidth / delta_f

    peaks, properties = sci_find_peaks(windowed_power, noise_median * (1.0 + noise_above), distance=max_width_bins, width=(min_width_bins, max_width_bins))

    detected_transmissions = [(windowed_freqs[idx], windowed_power[idx]) for idx in peaks if windowed_freqs[idx] > start_freq]
    center_frequencies = [point[0] for point in detected_transmissions]

    logger.info(f"Threshold: {noise_median * (1 + noise_above)}")

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.plot(windowed_freqs, windowed_power, label="Combined Spectrum")
        if detected_transmissions:
            trans_np = np.array(detected_transmissions)
            plt.plot(trans_np[:, 0], trans_np[:, 1], 'ro', markersize=8, label="Candidate Peaks")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title(f"Combined Spectrum from {start_freq / 1e6:.2f} MHz to {stop_freq / 1e6:.2f} MHz")
        plt.legend()
        plt.grid(True)
        plt.show()

    return center_frequencies


def build_spectrogram(candidates: list[float], sample_rate: float, num_samples: int):
    i = 0
    for freq in candidates:
        scanner = HackRfScanner(float(freq), sample_rate, num_samples)
        scanner.start()
        scanner.set_center_freq(freq)
        scanner.wait()

        samples = scanner.get_samples()

        np.save(f'samples_spect/samples_{i}.npy', samples, allow_pickle=True)
        # samples = np.load(f'samples/samples_{i}.npy')
        i += 1

        scanner.stop()

        # M = int(np.floor(np.log2(0.005 * sample_rate)))
        M = int(sample_rate//100)
        w = nuttall(M//2)
        SFT = ShortTimeFFT(w, hop=M//4, fs=sample_rate, fft_mode='centered')

        Sx = SFT.spectrogram(samples)
        del samples

        duration = num_samples/sample_rate

        f = np.linspace(freq-(sample_rate/2), freq+(sample_rate/2), Sx.shape[0])
        t = np.linspace(0, duration, Sx.shape[1])

        Sx_db = 10 * np.log10(np.fmax(Sx, 1e-4))
        del Sx

        plt.figure()
        plt.pcolormesh(t, f, Sx_db, shading='gouraud')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(f'spectrogram of {freq}')
        plt.colorbar(label='Power db')
        plt.show()

        del Sx_db




def main():
    start_freq = 88e6
    stop_freq = 110e6
    step = 200e3
    sample_rate = 8e6
    num_samples = int(sample_rate // 10)

    freq_candidates = scan_frequency_range(
                                    start_freq,
                                    stop_freq,
                                    step,
                                    sample_rate,
                                    num_samples,
                                    5e3,
                                    300e3,
                                    visualize=True,
                                    noise_above=0.03
                                   )

    logger.debug('Detected frequencies', detected=freq_candidates, num_detected=len(freq_candidates))

    # freq_candidates = [96.4e6]

    build_spectrogram(freq_candidates, 2**19, int(2**18))

if __name__ == '__main__':
    main()
