import serial
import time
import math
import os

# ------------------------
# CONFIGURATION
# ------------------------
PORT = 'COM3'
BAUD = 9600
FILENAME = "mpu_data.csv"
SAMPLES_PER_SECOND = 100  # 100 Hz
INTERVAL = 1 / SAMPLES_PER_SECOND

# ------------------------
# OPEN SERIAL
# ------------------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

# ------------------------
# OPEN CSV FILE
# ------------------------
file_exists = os.path.isfile(FILENAME)
f = open(FILENAME, "a")
if not file_exists:
    f.write("elapsed_time,ax,ay,az,vibration\n")

# ------------------------
# TIMER START
# ------------------------
start_time = time.time()
next_sample_time = time.perf_counter()

print("Logging started... Press Ctrl+C to stop.")

# ------------------------
# MAIN LOOP
# ------------------------
try:
    while True:
        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 3:
            continue

        try:
            ax = float(parts[0])
            ay = float(parts[1])
            az = float(parts[2])

            magnitude = math.sqrt(ax**2 + ay**2 + az**2)
            vibration = abs(magnitude - 16384)

            # --- elapsed time formatting ---
            elapsed_ms = int((time.time() - start_time) * 1000)
            h = elapsed_ms // 3600000
            m = (elapsed_ms % 3600000) // 60000
            s = (elapsed_ms % 60000) // 1000
            ms = elapsed_ms % 1000
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

            # Write to CSV
            f.write(f"{elapsed_str},{ax},{ay},{az},{vibration:.2f}\n")
            f.flush()

            # Print live data
            print(f"{elapsed_str},{ax},{ay},{az},{vibration:.2f}")

            # ---- precise 100 Hz timing ----
            next_sample_time += INTERVAL
            sleep_time = next_sample_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        except ValueError:
            continue

except KeyboardInterrupt:
    print("\nLogging stopped by user.")

finally:
    f.close()
    ser.close()
