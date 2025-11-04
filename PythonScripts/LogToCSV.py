import serial
import csv
from datetime import datetime

# ======== Config ========
PORT = "COM5"
BAUDRATE = 115200
SAVE_FOLDER = "./"
# ======================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_FOLDER}BNO055Data_{timestamp}.csv"

    print(f"Connecting to {PORT} at {BAUDRATE} baud...")
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)

    print(f"Logging data to {filename}")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

        try:
            while True:
                line = ser.readline().decode("utf-8").strip()
                if not line:
                    continue

                values = line.split(",")
                if len(values) == 6:
                    writer.writerow(values)
                    print(values)  # Displaying
        except KeyboardInterrupt:
            print("\nLogging stopped by user.")
        finally:
            ser.close()

if __name__ == "__main__":
    main()
