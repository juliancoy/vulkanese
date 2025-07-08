import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window  # Currently unused

# Modify sys.path to include necessary parent directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))
import vulkanese as ve
import vulkan as vk

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "sinode")))

if __name__ == "__main__":
    # Generate a harmonic-rich signal at 440Hz (A4)
    sr = 48000
    A4 = 440
    t = np.arange(2**15)
    z = (
        np.sin(t * 2 * np.pi * A4 / sr)
        + np.sin(2 * t * 2 * np.pi * A4 / sr)
        + np.sin(3 * t * 2 * np.pi * A4 / sr)
        + np.sin(4 * t * 2 * np.pi * A4 / sr)
    )

    # Setup parameters
    multiple = 40
    normalizedStep = 5.0 / sr
    fprime = np.arange(100 / sr, 3000 / sr, normalizedStep)

    # Initialize Loiacono (CPU-based reference)
    linst = ve.math.signals.loiacono.Loiacono(
        fprime=fprime, multiple=multiple, dtftlen=2**15
    )

    print("--- Running CPU Test ---")
    for i in range(10):
        spectrum = linst.run(z)

    readstart = time.time()
    print("Readtime " + str(time.time() - readstart))

    # Optional: Plot the spectrum
    graph = True
    if graph:
        fig, ax2 = plt.subplots()
        ax2.plot(linst.fprime * sr, spectrum)
        ax2.set_title("CPU Result")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        plt.show()
