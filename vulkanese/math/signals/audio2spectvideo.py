import numpy as np
import cv2
import librosa
import os
import sys
import time
from scipy.signal import find_peaks
import tqdm

# === Setup Import Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))
import vulkanese as ve  # Uses Loiacono transform
import vulkan as vk

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "sinode")))

# === PARAMETERS ===
AUDIO_FILE = sys.argv[1]
FPS = 60
BUFFER_FRAMES = 120  # Number of time steps to show in the video
NUM_BINS = 512  # Number of frequency bins
vertstretch = 6
horizstretch = 0.25
VIDEO_SIZE = (
    BUFFER_FRAMES * vertstretch,
    int(NUM_BINS * horizstretch),
)  # width x height (corrected to match buffer dimensions)
VIDEO_FILE = AUDIO_FILE.replace("mp3", "mp4").replace(".wav", ".mp4")
BG_COLOR = (0x00, 0x2A, 0x61)  # OpenCV uses BGR; this is #002a61 in BGR reversed
BG_COLOR = tuple(reversed(BG_COLOR))  # Convert to RGB
CURRENT_TIME_COLOR = (0, 0, 255)  # Red for current time line

# === LOAD AUDIO ===
y, sr = librosa.load(AUDIO_FILE, sr=None)
frame_duration = 1.0 / FPS
hop_size = int(frame_duration * sr)  # Audio samples per video frame

multiple = 15
# === SETUP Loiacono Transform ===
minfreq = 100
maxfreq = 12000
fprime = np.logspace(np.log10(minfreq / sr), np.log10(maxfreq / sr), NUM_BINS)
max_samples_per_period = multiple * sr / minfreq
dtftlen = 2 ** int(np.ceil(np.log2(max_samples_per_period)))  # FFT length
print(f"Using FFT size of {dtftlen} samples (original hop size: {hop_size})")

loiacono = ve.math.signals.loiacono.Loiacono(
    fprime=fprime, multiple=multiple, dtftlen=dtftlen
)

# === INIT BUFFER ===
buffer = np.zeros(
    (BUFFER_FRAMES, NUM_BINS), dtype=np.float32
)  # Now using float for gradient values

# === INIT VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, VIDEO_SIZE, isColor=True)

# === CONFIGURABLE OPTIONS ===
realtime_display = "--realtime" in sys.argv

# === MAIN ANALYSIS LOOP ===
num_frames = len(y) // hop_size
print(f"Processing {num_frames} frames with Loiacono transform...")

# Convert background color to numpy array for vector operations
bg_color_np = np.array(BG_COLOR, dtype=np.float32)
white_color = np.array([255, 255, 255], dtype=np.float32)
red_color = np.array([0, 0, 255], dtype=np.float32)  # Red color for middle tenth

# Initialize a window to center the display
if realtime_display:
    cv2.namedWindow("Spectral Peaks", cv2.WINDOW_NORMAL)
    screen_width = cv2.getWindowImageRect("Spectral Peaks")[2]

# Calculate which time rows are in the middle tenth (vertically)
middle_start = BUFFER_FRAMES // 2 - BUFFER_FRAMES // 20  # Half of tenth on each side
middle_end = BUFFER_FRAMES // 2 + BUFFER_FRAMES // 20

for i in tqdm.tqdm(range(num_frames)):
    start = i * hop_size
    end = start + dtftlen
    frame = y[start:end] if end <= len(y) else np.zeros(dtftlen)

    # Compute spectrum
    spectrum = loiacono.run(frame)

    # === Adaptive thresholding ===
    mean_val = np.mean(spectrum)
    std_val = np.std(spectrum)
    threshold = mean_val + 1.5 * std_val

    # Normalize spectrum values above threshold to [0, 1] range
    normalized = np.clip((spectrum - threshold) / (np.max(spectrum) - threshold), 0, 1)

    # Update rolling buffer - centered around current time
    buffer = np.roll(buffer, -1, axis=0)
    buffer[-1] = normalized  # Store normalized values for gradient

    # Create color frame with gradient
    img = np.zeros((BUFFER_FRAMES, NUM_BINS, 3), dtype=np.uint8)

    # Create full image with white gradient first
    buffer_3d = np.repeat(buffer[:, :, np.newaxis], 3, axis=2)
    img = (bg_color_np * (1 - buffer_3d) + white_color * buffer_3d)
    
    # Overwrite the middle tenth with red gradient
    img[middle_start:middle_end, :] = (
        bg_color_np * (1 - buffer_3d[middle_start:middle_end, :]) + 
        red_color * buffer_3d[middle_start:middle_end, :]
    )
    
    img = img.astype(np.uint8)

    # Rotate counterclockwise 90 degrees (270 degrees clockwise)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Stretch vertically using resize with nearest neighbor interpolation
    img = cv2.resize(img, VIDEO_SIZE, interpolation=cv2.INTER_NEAREST)

    # Add red line at center (current time)
    #center_pos = VIDEO_SIZE[0] // 2
    #img[:, center_pos-1 : center_pos + 1] = CURRENT_TIME_COLOR

    out.write(img)

    # Optional real-time display
    if realtime_display:
        # Center the window on screen
        if screen_width > 0:
            cv2.moveWindow("Spectral Peaks", (screen_width - NUM_BINS) // 2, 100)

        cv2.imshow("Spectral Peaks", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Real-time display interrupted by user.")
            break

out.release()
cv2.destroyAllWindows()
print(f"Video written to {VIDEO_FILE}")