import serial

ser = serial.Serial('COM3', 9600, timeout=1)

print("Listening to COM3...")

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(line)
    except Exception as e:
        print("Error:", e)
