"""
Device Manager - Manages Android device connections.
"""

from typing import Dict, List, Optional

from droidrun.adb.device import Device
from droidrun.adb.wrapper import ADBWrapper

import logging


logger = logging.getLogger("device_manager")
logger.level = logging.DEBUG


def setup_logging(quiet: bool = False):
    """Setup logging for Device Management"""
    if quiet:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="🎩 %(message)s")


setup_logging(False)


class DeviceManager:
    """Manages Android device connections."""

    def __init__(self, adb_path: Optional[str] = None):
        """Initialize device manager.

        Args:
            adb_path: Path to ADB binary
        """
        self._adb = ADBWrapper(adb_path)
        self._devices: Dict[str, Device] = {}

    async def list_devices(self) -> List[Device]:
        """List connected devices.

        Returns:
            List of connected devices
        """
        logger.info("Refreshing the device list")
        devices_info = await self._adb.get_devices()

        # Update device cache
        current_serials = set()
        for device_info in devices_info:
            serial = device_info["serial"]
            current_serials.add(serial)

            if serial not in self._devices:
                self._devices[serial] = Device(serial, self._adb)

        # Remove disconnected devices
        for serial in list(self._devices.keys()):
            if serial not in current_serials:
                del self._devices[serial]

        return list(self._devices.values())

    async def get_device(self, serial: str) -> Optional[Device]:
        """Get a specific device.

        Args:
            serial: Device serial number

        Returns:
            Device instance if found, None otherwise
        """
        if serial in self._devices:
            return self._devices[serial]
        else:
            # perhaps the cache hasn't been refreshed, CHECK!
            updated_list = await self.list_devices()

        # now you can try again
        # note this happens the first time you query a device from the
        # device manager, because it does not fetch the full list of devices
        # the first time, because that would require an async call in the init.

        if serial in self._devices:
            return self._devices[serial]

        if serial not in updated_list:
            return None

    async def connect(self, host: str, port: int = 5555) -> Optional[Device]:
        """Connect to a device over TCP/IP.

        Args:
            host: Device IP address
            port: Device port

        Returns:
            Connected device instance
        """
        try:
            serial = await self._adb.connect(host, port)
            return await self.get_device(serial)
        except Exception:
            return None

    async def disconnect(self, serial: str) -> bool:
        """Disconnect from a device.

        Args:
            serial: Device serial number

        Returns:
            True if disconnected successfully
        """
        success = await self._adb.disconnect(serial)
        if success and serial in self._devices:
            del self._devices[serial]
        return success
