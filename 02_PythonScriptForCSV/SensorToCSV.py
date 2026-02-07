import serial
import csv
import time
from datetime import datetime

PORT = 'COM3'      # change if needed
BAUD = 9600
FILENAME = "sensor_data.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

file = open(FILENAME, mode='w', newline='', buffering=1)
writer = csv.writer(file)

# Header
writer.writerow(["time", "ax", "ay", "az", "gx", "gy", "gz", "sw420"])
file.flush()

print("Logging started... Press CTRL+C to stop")

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        data = line.split(",")

        if len(data) == 7:
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S") + f":{int(now.microsecond/1000):03d}"

            writer.writerow([timestamp] + data)
            file.flush()   # 🔥 FORCE WRITE TO DISK
            print(timestamp, data)

except KeyboardInterrupt:
    print("\nLogging stopped")

finally:
    file.close()
    ser.close()
