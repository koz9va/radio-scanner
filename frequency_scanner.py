import osmosdr
from math import floor
import structlog
import numpy as np
from gnuradio import gr, blocks, analog
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as sci_find_peaks,  savgol_filter
import cupyx.scipy.signal as cu_signal
import cupy

structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()


def median_truncate(data: np.ndarray, div_factor: int) -> np.ndarray:
    out = np.zeros(len(data) // div_factor, dtype=data.dtype)
    sg_size = int(len(data) / div_factor)  # segment size
    for i in range(len(out) - 1):
        out[i:i + sg_size] = np.median(data[i:i + sg_size])

    return out



class RfScanner(gr.top_block):
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
        # self.source.set_gain(20)
        self.source.set_if_gain(16) # default from official docs. No more than 47
        self.source.set_bb_gain(16)
        self.source.set_gain(0)

        # self.agc = analog.agc_cc(1e-3, 1.0, 1.0)
        # self.agc.set_max_gain(1000)


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

        self.connect(self.source, self.head, self.sink)

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
        scanner = RfScanner(float(start_freq), sample_rate, num_samples)
        scanner.start()
        scanner.set_center_freq(freq)
        scanner.wait()

        samples = scanner.get_samples()
        #
        # np.save(f'samples/samples_{i}.npy', samples, allow_pickle=True)
        # samples = np.load(f'samples/samples_{i}.npy')
        # i += 1

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
    # windowed_power = all_power_sorted.reshape(-1, 1000).mean(axis=1)
    windowed_power = savgol_filter(all_power_sorted, window_length=101, polyorder=4)
    windowed_power = windowed_power[::1000]
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
        plt.savefig('samples/images/all_spectrum.png')

    return center_frequencies



def downsample(samples, factor):
    return samples[::factor]


def get_noise_threshold(s_mag: np.ndarray, noise_additive = 3):
    flat_db = s_mag.flatten()
    hist_vals, bin_edges = np.histogram(flat_db, bins=200)
    noise_floor = bin_edges[np.argmax(hist_vals)]
    return noise_floor + noise_additive # db


def build_spectrogram(candidates: list[float], sample_rate: float, num_samples: int):
    downsample_factor = 10
    for freq in candidates:
        scanner = RfScanner(float(freq), sample_rate, num_samples)
        scanner.start()
        scanner.set_center_freq(freq)
        scanner.wait()
        samples = cupy.asarray(scanner.get_samples())
        scanner.stop()


        zoomed_samples = downsample(samples, downsample_factor)

        sample_rate /= downsample_factor

        S_complex = cu_signal.stft(zoomed_samples, fs=sample_rate)[2]

        # complex spectrogram
        # S_complex = SFT.stft(samples)
        del samples

        # axes
        n_freq, n_time = S_complex.shape
        duration = num_samples / sample_rate
        f = np.linspace(freq - sample_rate/2, freq + sample_rate/2, n_freq)
        t = np.linspace(0, duration, n_time)

        # 1) magnitude (dB)

        s_mag_db_gpu = 20 * cupy.log10(np.abs(S_complex) + 1e-12)

        noise_level = get_noise_threshold(s_mag_db_gpu, noise_additive=2)

        s_mag_db_gpu = s_mag_db_gpu / (1 + cupy.exp(0.5 * (s_mag_db_gpu - noise_level)))

        s_mag_db = cupy.asnumpy(s_mag_db_gpu)

        plt.figure(figsize=(8,4))
        plt.pcolormesh(t, f, s_mag_db, shading='gouraud')
        plt.title(f'Magnitude Spectrogram ({freq/1e6:.3f} MHz)')
        plt.xlabel('Time [s]'); plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.savefig(f'samples/images/spec_{floor(freq/1e3)}_mag.png')
        plt.close()

        # 2) phase
        # phase_spec = np.angle(cupy.asnumpy(S_complex))
        # plt.figure(figsize=(8,4))
        # plt.pcolormesh(t, f, phase_spec, shading='gouraud', cmap='twilight')
        # plt.title(f'Phase Spectrogram ({freq/1e6:.3f} MHz)')
        # plt.xlabel('Time [s]'); plt.ylabel('Frequency [Hz]')
        # plt.colorbar(label='rad')
        # plt.tight_layout()
        # plt.savefig(f'samples/images/spec_{freq/1e3}_phase.png')
        # plt.close()

        # 3) instantaneous frequency
        # unwrapped = np.unwrap(phase_spec, axis=1)
        # dphase   = np.diff(unwrapped, axis=1)
        # dt       = hop / sample_rate
        # inst_freq = (dphase/(2*np.pi)) / dt
        # t_if = t[:-1]
        # plt.figure(figsize=(8,4))
        # plt.pcolormesh(t_if, f, inst_freq, shading='gouraud', cmap='magma')
        # plt.title(f'Instantaneous Frequency ({freq/1e6:.3f} MHz)')
        # plt.xlabel('Time [s]'); plt.ylabel('Frequency [Hz]')
        # plt.colorbar(label='Hz')
        # plt.tight_layout()
        # plt.savefig(f'samples/images/spec_{round(freq/1e6)}_instfreq.png')
        # plt.close()


def main():
    start_freq = 98e6
    stop_freq = 101e6
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

    # freq_candidates = [e6]
    # #
    # build_spectrogram(freq_candidates, 2**25, int(2**20))
    build_spectrogram(freq_candidates, 2**23, int(2**20))

if __name__ == '__main__':
    main()
