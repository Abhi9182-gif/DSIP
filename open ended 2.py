import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INPUT_FILE = "input_song.wav"
OUTPUT_DIR = "audio_results"

ALPHA = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# LOAD AUDIO
# ---------------------------------------------------------

def load_audio(filename):

    fs, audio = wavfile.read(filename)

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Convert to floating point
    audio = audio.astype(np.float64)

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value != 0:
        audio = audio / max_value

    return fs, audio


# ---------------------------------------------------------
# SAVE AUDIO
# ---------------------------------------------------------

def save_audio(filename, fs, audio):

    audio = np.asarray(audio)

    max_value = np.max(np.abs(audio))

    if max_value != 0:
        audio = audio / max_value

    audio = np.int16(audio * 32767)

    wavfile.write(filename, fs, audio)


# ---------------------------------------------------------
# IMPULSE RESPONSES
# ---------------------------------------------------------

# 1. Identity impulse
def identity_ir():

    h = np.array([1.0])

    return h


# 2. Moving average impulse response
def moving_average_ir():

    h = np.ones(9)

    h = h / np.sum(h)

    return h


# 3. Alternating impulse response
def alternating_ir():

    h = np.array([
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1
    ], dtype=float)

    h = h / np.sum(np.abs(h))

    return h


# 4. Exponential decay impulse response
def exponential_ir():

    n = np.arange(32)

    h = np.exp(-0.18 * n)

    h = h / np.sum(np.abs(h))

    return h


# 5. Echo impulse response
def echo_ir():

    delay = 12
    gain = 0.45

    h = np.zeros(delay + 1)

    h[0] = 1
    h[delay] = gain

    h = h / np.sum(np.abs(h))

    return h


# ---------------------------------------------------------
# CONVOLUTION
# ---------------------------------------------------------

def convolution(x, h):

    y = fftconvolve(x, h, mode="same")

    return y


# ---------------------------------------------------------
# INVERSE FILTERING
# ---------------------------------------------------------

def inverse_filter(y, h):

    N = len(y)

    nfft = N + len(h) - 1

    # FFT of signal
    Y = np.fft.rfft(y, n=nfft)

    # FFT of impulse response
    H = np.fft.rfft(h, n=nfft)

    # Regularized inverse filter
    H_inverse = np.conj(H) / (
        np.abs(H) ** 2 + ALPHA
    )

    # Apply inverse filter
    X_estimated = np.fft.irfft(
        Y * H_inverse,
        n=nfft
    )

    # Select original length
    start = (len(h) - 1) // 2

    X_estimated = X_estimated[
        start:start + N
    ]

    return X_estimated


# ---------------------------------------------------------
# RMS
# ---------------------------------------------------------

def calculate_rms(x):

    return np.sqrt(
        np.mean(x ** 2)
    )


# ---------------------------------------------------------
# CORRELATION
# ---------------------------------------------------------

def calculate_correlation(x, y):

    x = x - np.mean(x)
    y = y - np.mean(y)

    numerator = np.sum(x * y)

    denominator = np.sqrt(
        np.sum(x ** 2) *
        np.sum(y ** 2)
    )

    if denominator == 0:
        return 0

    return numerator / denominator


# ---------------------------------------------------------
# SPECTRAL CENTROID
# ---------------------------------------------------------

def spectral_centroid(x, fs):

    X = np.abs(
        np.fft.rfft(x)
    )

    frequencies = np.fft.rfftfreq(
        len(x),
        1 / fs
    )

    denominator = np.sum(X)

    if denominator == 0:
        return 0

    centroid = np.sum(
        frequencies * X
    ) / denominator

    return centroid


# ---------------------------------------------------------
# PLOT RESULTS
# ---------------------------------------------------------

def plot_results(
        original,
        convolved,
        recovered,
        h,
        fs,
        name):

    # Show first 5 seconds
    number_of_samples = min(
        len(original),
        int(5 * fs)
    )

    time = np.arange(
        number_of_samples
    ) / fs

    # -----------------------------
    # Waveform plot
    # -----------------------------

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)

    plt.plot(
        time,
        original[:number_of_samples]
    )

    plt.title(
        "Original Audio"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid()


    plt.subplot(3, 1, 2)

    plt.plot(
        time,
        convolved[:number_of_samples]
    )

    plt.title(
        "After Convolution - " + name
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid()


    plt.subplot(3, 1, 3)

    plt.plot(
        time,
        recovered[:number_of_samples]
    )

    plt.title(
        "After Inverse Filtering - " + name
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            name + "_comparison.png"
        ),
        dpi=200
    )

    plt.show()


    # -----------------------------
    # Impulse response plot
    # -----------------------------

    plt.figure(figsize=(10, 4))

    plt.stem(
        np.arange(len(h)),
        h
    )

    plt.title(
        "Impulse Response - " + name
    )

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            name + "_IR.png"
        ),
        dpi=200
    )

    plt.show()


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    # Check input file

    if not os.path.exists(INPUT_FILE):

        print(
            "ERROR: input_song.wav not found!"
        )

        print(
            "Place input_song.wav in the same folder "
            "as this Python program."
        )

        return


    # Load audio

    fs, original = load_audio(
        INPUT_FILE
    )

    print("\n")
    print("=" * 60)
    print(
        "AUDIO CONVOLUTION AND INVERSE FILTERING"
    )
    print("=" * 60)

    print(
        "Sampling Frequency:",
        fs,
        "Hz"
    )

    print(
        "Number of Samples:",
        len(original)
    )

    print(
        "Original RMS:",
        calculate_rms(original)
    )

    print("\n")


    # Create impulse responses

    impulse_responses = {

        "IR1_Identity":
            identity_ir(),

        "IR2_Moving_Average":
            moving_average_ir(),

        "IR3_Alternating":
            alternating_ir(),

        "IR4_Exponential":
            exponential_ir(),

        "IR5_Echo":
            echo_ir()
    }


    results = []


    # Process each impulse response

    for name, h in impulse_responses.items():

        print(
            "Processing:",
            name
        )


        # Convolution

        convolved = convolution(
            original,
            h
        )


        # Inverse filtering

        recovered = inverse_filter(
            convolved,
            h
        )


        # Save convolved audio

        save_audio(
            os.path.join(
                OUTPUT_DIR,
                name +
                "_convolved.wav"
            ),
            fs,
            convolved
        )


        # Save inverse filtered audio

        save_audio(
            os.path.join(
                OUTPUT_DIR,
                name +
                "_inverse_filtered.wav"
            ),
            fs,
            recovered
        )


        # Calculate measurements

        original_rms = calculate_rms(
            original
        )

        convolved_rms = calculate_rms(
            convolved
        )

        recovered_rms = calculate_rms(
            recovered
        )

        correlation = calculate_correlation(
            original,
            recovered
        )

        original_centroid = spectral_centroid(
            original,
            fs
        )

        recovered_centroid = spectral_centroid(
            recovered,
            fs
        )


        results.append([
            name,
            len(h),
            original_rms,
            convolved_rms,
            recovered_rms,
            correlation,
            original_centroid,
            recovered_centroid
        ])


        # Print results

        print(
            "IR Length:",
            len(h)
        )

        print(
            "Convolved RMS:",
            convolved_rms
        )

        print(
            "Recovered RMS:",
            recovered_rms
        )

        print(
            "Correlation:",
            correlation
        )

        print()


        # Generate plots

        plot_results(
            original,
            convolved,
            recovered,
            h,
            fs,
            name
        )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results_file = os.path.join(
        OUTPUT_DIR,
        "experiment_results.csv"
    )


    with open(
        results_file,
        "w"
    ) as file:

        file.write(
            "IR,IR_Length,"
            "Original_RMS,"
            "Convolved_RMS,"
            "Recovered_RMS,"
            "Correlation,"
            "Original_Centroid,"
            "Recovered_Centroid\n"
        )


        for row in results:

            file.write(
                ",".join(
                    map(str, row)
                )
                + "\n"
            )


    print("=" * 60)

    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print(
        "Results saved in:",
        OUTPUT_DIR
    )


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":

    main()