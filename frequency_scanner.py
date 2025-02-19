import osmosdr
import structlog
import numpy as np
from gnuradio import gr, blocks, analog
from scipy.signal import find_peaks as sci_find_peaks
import matplotlib.pyplot as plt

structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

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
                         visualize: bool = True) -> list[float]:
    # detected_transmissions = []
    all_freqs_list = []
    all_power_list = []
    delta_f = sample_rate / num_samples
    min_width_bins = int(min_bandwidth / delta_f)
    max_width_bins = int(max_bandwidth / delta_f)
    for freq in np.arange(start_freq, stop_freq, step):
        scanner = HackRfScanner(float(start_freq), sample_rate, num_samples)
        scanner.start()
        scanner.set_center_freq(freq)
        scanner.wait()

        samples = scanner.get_samples()

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

    windowed_freqs = all_freqs_sorted.reshape(-1, len(all_freqs_sorted) // 1000).mean(axis=1)
    windowed_power = all_power_sorted.reshape(-1, len(all_power_sorted) // 1000).mean(axis=1)

    peaks, properties = sci_find_peaks(windowed_power, noise_median * 1.05)

    detected_transmissions = [(windowed_freqs[idx], windowed_power[idx]) for idx in peaks]
    center_frequencies = [point[0] for point in detected_transmissions]

    logger.info(f"Threshold: {noise_median * 1.05}")

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


def main():
    start_freq = 95e6
    stop_freq = 100e6
    step = 200e3
    sample_rate = 16e6
    num_samples = int(sample_rate // 10)

    freq_candidates = scan_frequency_range(start_freq,
                                    stop_freq,
                                    step,
                                    sample_rate,
                                    num_samples,
                                    1e3,
                                    250e3,
                                    visualize=True)
    logger.debug('Detected frequencies', detected=freq_candidates, num_detected=len(freq_candidates))


if __name__ == '__main__':
    main()
