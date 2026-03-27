import os
import random
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = "mobile_phone_charger_data.csv"

NUM_CAPTURES = 120              # number of rows
SAMPLE_RATE = 2000              # Hz
DURATION = 0.25                 # seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

RANDOM_SEED = 42

DEVICE_TYPE = "usb_phone_charger"
DEVICE_STATES = ["idle", "charging"]

DISTANCES = [2, 5, 10, 15]      # cm
ORIENTATIONS = [0, 30, 60, 90]  # degrees
GAINS = [1, 2, 4]

# ============================================================
# SET SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# TIME AXIS
# ============================================================

def create_time_axis():
    return np.arange(NUM_SAMPLES) / SAMPLE_RATE


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_charger_signal(t, state):
    """
    Create a realistic-ish phone charger EMI signal.

    Key characteristics:
    - switching frequency (dominant)
    - harmonics
    - slight modulation (not perfectly clean)
    """

    if state == "idle":
        base_freq = random.uniform(150, 220)
        amplitude = random.uniform(0.2, 0.35)
    else:  # charging
        base_freq = random.uniform(180, 260)
        amplitude = random.uniform(0.35, 0.6)

    # Base + harmonics
    signal = (
        amplitude * np.sin(2 * np.pi * base_freq * t) +
        0.3 * amplitude * np.sin(2 * np.pi * 2 * base_freq * t) +
        0.15 * amplitude * np.sin(2 * np.pi * 3 * base_freq * t)
    )

    # Add small modulation (real signals are not perfect)
    mod_freq = random.uniform(2, 8)
    modulation = 1 + 0.05 * np.sin(2 * np.pi * mod_freq * t)
    signal = signal * modulation

    return signal


# ============================================================
# REAL-WORLD EFFECTS
# ============================================================

def apply_distance(signal, distance):
    """
    Signal weakens with distance.
    """
    scale = 1 / (1 + 0.1 * distance)
    return signal * scale


def apply_orientation(signal, orientation):
    """
    Orientation affects signal strength.
    """
    factor = 0.3 + 0.7 * abs(np.cos(np.radians(orientation)))
    return signal * factor


def apply_gain(signal, gain):
    """
    Amplification.
    """
    return signal * gain


def add_noise(signal, state):
    """
    Add noise to simulate real environment.
    """
    if state == "idle":
        noise_std = 0.05
    else:
        noise_std = 0.07

    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise


def add_background_interference(signal, t):
    """
    Add small random frequency interference.
    """
    for _ in range(2):
        freq = random.uniform(40, 500)
        amp = random.uniform(0.01, 0.05)
        signal += amp * np.sin(2 * np.pi * freq * t)

    return signal


# ============================================================
# ONE CAPTURE
# ============================================================

def generate_one_capture(trial_id, t):
    """
    Generate one row of data.
    """

    state = random.choice(DEVICE_STATES)
    distance = random.choice(DISTANCES)
    orientation = random.choice(ORIENTATIONS)
    gain = random.choice(GAINS)

    # Base signal
    signal = generate_charger_signal(t, state)

    # Apply effects
    signal = apply_distance(signal, distance)
    signal = apply_orientation(signal, orientation)
    signal = apply_gain(signal, gain)

    # Add noise + interference
    signal = add_background_interference(signal, t)
    signal = add_noise(signal, state)

    # Create row
    row = {
        "trial_id": trial_id,
        "device_type": DEVICE_TYPE,
        "device_state": state,
        "distance_cm": distance,
        "orientation_deg": orientation,
        "gain": gain,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION
    }

    # Add waveform samples
    for i, val in enumerate(signal):
        row[f"sample_{i}"] = float(val)

    return row


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():
    """
    Generate full dataset.
    """
    t = create_time_axis()
    rows = []

    for i in range(1, NUM_CAPTURES + 1):
        rows.append(generate_one_capture(i, t))

    df = pd.DataFrame(rows)
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("Generating mobile phone charger dataset...")

    df = generate_dataset()

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()