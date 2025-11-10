import asyncio
from bleak import BleakClient, BleakScanner
import csv
from datetime import datetime

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef1"

SAVE_FOLDER = "./"

async def main():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    target = None

    for d in devices:
        if d.name and "BNO055_BLE_Sensor" in d.name:
            target = d
            break

    if target is None:
        print("Device not found.")
        print("=== Available Devices ===")
        for d in devices:
            print(f"- {d.name} ({d.address})")
        print("=========================")
        return

    print(f"Connecting to {target.name} ({target.address})...")
    async with BleakClient(target.address) as client:
        connected = client.is_connected
        print("Connected:", connected)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_FOLDER}BNO055Data_{timestamp}.csv"
        f = open(filename, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

        def notification_handler(sender, data):
            line = data.decode("utf-8").strip()
            values = line.split(",")
            if len(values) == 6:
                writer.writerow(values)
                print(values)

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        print("Receiving data... Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await client.stop_notify(CHARACTERISTIC_UUID)
            f.close()

if __name__ == "__main__":
    asyncio.run(main())
