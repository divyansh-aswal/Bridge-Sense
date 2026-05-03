"""
Cantilever Beam SHM — Python Data Logger
=========================================
- Reads 100Hz data from Arduino over serial
- Logs to CSV with proper timestamp (HH:MM:SS:mmm)
- Adds damage_level label at collection time
- Verifies actual sampling rate in real time
- Shows live sample count and rate feedback
- Handles serial buffer overflow gracefully

Usage:
    python collect_data.py

You will be prompted to enter:
    - Damage level (0, 1, 2, or 3)
    - Session number (1, 2, or 3)
    - Duration in seconds

Output files:
    damage_L{level}_S{session}.csv   (one file per session)
"""

import serial
import csv
import time
import sys
from datetime import datetime

# ── CONFIG — change only these ────────────────────────────────────────────────
PORT         = 'COM3'       # Change to your port (check Arduino IDE bottom right)
                            # On Mac/Linux: '/dev/ttyUSB0' or '/dev/ttyACM0'
BAUD         = 115200       # Must match Arduino Serial.begin(115200)
TARGET_HZ    = 100          # Expected sample rate
GRAVITY_AXIS = 'az'         # Which axis has gravity (usually az for flat sensor)
# ─────────────────────────────────────────────────────────────────────────────

def get_session_info():
    """Ask user for damage level and session number before recording."""
    print("\n" + "="*50)
    print("  CANTILEVER BEAM DATA COLLECTOR")
    print("="*50)
    print("\nDamage Levels:")
    print("  0 = Healthy (no notch)")
    print("  1 = Light damage (~10-15% notch)")
    print("  2 = Moderate damage (~25-30% notch)")
    print("  3 = Severe damage (~40-50% notch)")

    while True:
        try:
            level = int(input("\nEnter damage level (0/1/2/3): "))
            if level in [0, 1, 2, 3]:
                break
            print("  Please enter 0, 1, 2, or 3")
        except ValueError:
            print("  Please enter a number")

    while True:
        try:
            session = int(input("Enter session number (1/2/3): "))
            if session in [1, 2, 3]:
                break
            print("  Please enter 1, 2, or 3")
        except ValueError:
            print("  Please enter a number")

    while True:
        try:
            duration = int(input("Recording duration in seconds (recommended: 120): "))
            if duration > 0:
                break
        except ValueError:
            print("  Please enter a positive number")

    label_map = {0: "Healthy", 1: "Light_Damage", 2: "Moderate_Damage", 3: "Severe_Damage"}
    filename = f"damage_L{level}_S{session}.csv"

    return level, session, duration, filename, label_map[level]


def compute_gravity_offset(ser, n_samples=200):
    """
    Read n_samples at startup to compute gravity offset on az.
    This removes the ~15031 gravitational component from az.
    Hold the sensor STILL during this phase.
    """
    print(f"\nCalibrating gravity offset — hold beam COMPLETELY STILL for 2 seconds...")
    samples = []
    ser.reset_input_buffer()

    while len(samples) < n_samples:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or line.startswith('sample_id'):
            continue
        parts = line.split(',')
        if len(parts) == 8:
            try:
                az = float(parts[3])
                samples.append(az)
            except ValueError:
                continue

    gravity_offset = sum(samples) / len(samples)
    print(f"  Gravity offset (az mean over {n_samples} samples): {gravity_offset:.2f}")
    print(f"  This will be subtracted from all az readings as 'az_zeroed'")
    return gravity_offset


def collect_data(level, session, duration, filename, label, gravity_offset):
    """Main data collection loop."""

    print(f"\n{'='*50}")
    print(f"  Recording: Damage Level {level} ({label}), Session {session}")
    print(f"  Duration: {duration} seconds")
    print(f"  Output: {filename}")
    print(f"{'='*50}")
    print("\nPress ENTER when ready to start recording...")
    input()

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # let Arduino reset and stabilize
    ser.reset_input_buffer()  # discard buffered data from reset

    file = open(filename, mode='w', newline='', buffering=1)
    writer = csv.writer(file)

    # ── CSV header ────────────────────────────────────────────────────────────
    writer.writerow([
        "sample_id", "timestamp",
        "damage_level", "damage_label",
        "session",
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "sw420",
        "az_zeroed"
    ])
    file.flush()

    print(f"Recording started — {duration}s at {TARGET_HZ}Hz...")
    print(f"Expected samples: {duration * TARGET_HZ:,}")
    print("Press CTRL+C to stop early\n")

    start_time   = time.time()
    end_time     = start_time + duration
    row_count    = 0
    last_report  = start_time
    last_rate_count = 0

    try:
        while time.time() < end_time:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            # Skip empty lines and header line from Arduino
            if not line or line.startswith('sample_id'):
                continue

            parts = line.split(',')
            if len(parts) != 8:
                continue  # malformed line — skip silently

            try:
                s_id = int(parts[0])
                ax   = int(parts[1])
                ay   = int(parts[2])
                az   = int(parts[3])
                gx   = int(parts[4])
                gy   = int(parts[5])
                gz   = int(parts[6])
                sw   = int(parts[7])
            except ValueError:
                continue  # non-numeric — skip

            # ── Timestamp (PC clock — accurate enough) ────────────────────────
            now = datetime.now()
            ms  = now.microsecond // 1000
            timestamp = now.strftime(f"%H:%M:%S:{ms:03d}")

            # ── az_zeroed — remove gravity ─────────────────────────────────────
            az_zeroed = round(az - gravity_offset, 4)

            writer.writerow([
                s_id, timestamp,
                level, label,
                f"L{level}_S{session}",
                ax, ay, az,
                gx, gy, gz,
                sw,
                az_zeroed
            ])

            row_count += 1

            # ── Live rate display every 5 seconds ─────────────────────────────
            now_t = time.time()
            if now_t - last_report >= 5.0:
                elapsed    = now_t - start_time
                remaining  = end_time - now_t
                rate       = (row_count - last_rate_count) / (now_t - last_report)
                last_rate_count = row_count
                last_report = now_t

                print(f"  [{elapsed:.0f}s elapsed | {remaining:.0f}s remaining] "
                      f"Samples: {row_count:,} | Rate: {rate:.1f} Hz", end='\r')

    except KeyboardInterrupt:
        print(f"\n\nStopped early by user at {row_count:,} samples")

    finally:
        file.flush()
        file.close()
        ser.close()

    # ── Final report ──────────────────────────────────────────────────────────
    actual_duration = time.time() - start_time
    actual_rate     = row_count / actual_duration

    print(f"\n{'='*50}")
    print(f"  RECORDING COMPLETE")
    print(f"{'='*50}")
    print(f"  File saved   : {filename}")
    print(f"  Samples      : {row_count:,}")
    print(f"  Duration     : {actual_duration:.1f}s")
    print(f"  Actual rate  : {actual_rate:.1f} Hz  (target: {TARGET_HZ} Hz)")

    if actual_rate < TARGET_HZ * 0.90:
        print(f"\n  WARNING: Actual rate ({actual_rate:.1f} Hz) is below 90% of target.")
        print(f"  Check: baud rate = 115200, Wire.setClock(400000) in Arduino code.")
    else:
        print(f"  Rate OK — within acceptable range.")

    print(f"\n  Next: run merge_sessions.py after all sessions are collected.")


def main():
    # ── Check port connection ─────────────────────────────────────────────────
    print(f"Connecting to Arduino on {PORT} at {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()

        # ── Gravity calibration ───────────────────────────────────────────────
        gravity_offset = compute_gravity_offset(ser)
        ser.close()

    except serial.SerialException as e:
        print(f"\nERROR: Cannot open {PORT}")
        print(f"  {e}")
        print(f"\nFix: Check your port in Arduino IDE → Tools → Port")
        sys.exit(1)

    # ── Session info ──────────────────────────────────────────────────────────
    level, session, duration, filename, label = get_session_info()

    # ── Reopen serial for recording ───────────────────────────────────────────
    collect_data(level, session, duration, filename, label, gravity_offset)


if __name__ == '__main__':
    main()