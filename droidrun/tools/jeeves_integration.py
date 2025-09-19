#!/usr/bin/env python3
"""
Jeeves Integration for DroidRun
Fast UI state access via Accessibility Service instead of uiautomator

This provides a drop-in replacement for uiautomator-based UI state retrieval
with significant performance and reliability improvements.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class JeevesIntegration:
    """
    Integration layer for Jeeves Android Accessibility Service
    
    Provides fast, reliable UI state access without uiautomator's concurrency issues.
    """

    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.jeeves_package = "com.jeeves"
        self.jeeves_service = "com.jeeves/.JeevesService"
        self.content_provider_uri = "content://com.jeeves.uistate/elements"
        self.stats_uri = "content://com.jeeves.uistate/stats"
        self.is_available = None  # Cache availability check

    async def is_jeeves_available(self, device_serial: Optional[str] = None) -> bool:
        """Check if Jeeves service is available and running."""
        if self.is_available is not None:
            return self.is_available

        try:
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()

            # Check if Jeeves APK is installed
            result = await device._adb.shell(device._serial, f"pm list packages {self.jeeves_package}")
            if self.jeeves_package not in result:
                logger.info("📦 Jeeves APK not installed")
                self.is_available = False
                return False

            # Check if accessibility service is enabled
            result = await device._adb.shell(
                device._serial,
                "settings get secure enabled_accessibility_services"
            )

            if self.jeeves_service not in result:
                logger.info("🔐 Jeeves accessibility service not enabled")
                # Try to enable it automatically (may require user interaction)
                await self._try_enable_accessibility_service(device)

                # Check again
                result = await device._adb.shell(
                    device._serial,
                    "settings get secure enabled_accessibility_services"
                )

                if self.jeeves_service not in result:
                    logger.warning("⚠️ Jeeves accessibility service could not be enabled automatically")
                    self.is_available = False
                    return False

            logger.info("✅ Jeeves service is available and enabled")
            self.is_available = True
            return True

        except Exception as e:
            logger.error(f"Error checking Jeeves availability: {e}")
            self.is_available = False
            return False

    async def _try_enable_accessibility_service(self, device):
        """Try to enable Jeeves accessibility service automatically."""
        try:
            # Start the main activity to trigger user awareness
            await device._adb.shell(
                device._serial,
                f"am start -n {self.jeeves_package}/.MainActivity"
            )

            # Try to enable the service (may require user interaction)
            settings_command = (
                f"settings put secure enabled_accessibility_services "
                f"{self.jeeves_service}"
            )
            await device._adb.shell(device._serial, settings_command)

            # Enable accessibility in general
            await device._adb.shell(
                device._serial,
                "settings put secure accessibility_enabled 1"
            )

            # Give it a moment to start
            await asyncio.sleep(2)

            logger.info("🔧 Attempted to enable Jeeves accessibility service")

        except Exception as e:
            logger.debug(f"Could not auto-enable accessibility service: {e}")

    def _parse_content_provider_response(self, response: str) -> Dict[str, Any]:
        """
        Parse ContentProvider response format and convert to UI state format.
        ContentProvider returns tabular data, we convert to JSON-like structure.
        """
        try:
            elements = []
            lines = response.strip().split('\n')

            # Skip header line if present
            data_lines = [line for line in lines if line.strip() and not line.startswith('Row:')]

            for i, line in enumerate(data_lines):
                if 'index=' in line:  # This is a data row
                    # Parse the row format: index=0, class=android.widget.Button, text=Click, etc.
                    element = {"index": i}

                    # Extract fields using simple parsing
                    parts = line.split(', ')
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            # Map to expected field names
                            if key == 'index':
                                element['index'] = int(value) if value.isdigit() else i
                            elif key == 'class':
                                element['class'] = value
                            elif key == 'text':
                                element['text'] = value
                            elif key == 'content_desc':
                                element['content_desc'] = value
                            elif key == 'resource_id':
                                element['resource_id'] = value
                            elif key == 'clickable':
                                element['clickable'] = value == '1'
                            elif key == 'enabled':
                                element['enabled'] = value == '1'
                            elif key in ['left', 'top', 'right', 'bottom']:
                                if 'bounds' not in element:
                                    element['bounds'] = {}
                                element['bounds'][key] = int(value) if value.isdigit() else 0

                    elements.append(element)

            return {
                "a11y_tree": elements,
                "phone_state": {"activity": "unknown"},
                "source": "jeeves_content_provider",
                "method": "direct_memory_access",
                "file_io": False
            }

        except Exception as e:
            logger.error(f"Error parsing ContentProvider response: {e}")
            return {
                "error": "ContentProvider Parse Error",
                "message": str(e),
                "a11y_tree": [],
                "phone_state": {"activity": "unknown"}
            }

    async def _get_ui_state_via_broadcast(self, device, start_time: float) -> Dict[str, Any]:
        """
        Get UI state via logcat broadcast - FASTEST method!
        No ContentProvider, no file I/O, just pure logcat data transfer
        """
        try:
            # Clear logcat for clean data
            await device._adb.shell(device._serial, "logcat -c")

            # Send the UI state request
            await device._adb.shell(
                device._serial,
                "am broadcast -a com.jeeves.GET_UI_STATE"
            )

            # Poll for the data with retries
            max_retries = 50  # Increased retries for robustness
            ui_data_lines = []

            for retry in range(max_retries):
                # Wait for Jeeves processing
                wait_time = 0.05 # Increased wait time
                await asyncio.sleep(wait_time)

                # Get logcat data
                logcat_result = await device._adb.shell(
                    device._serial,
                    "logcat -d"
                )
                logger.debug(f"Raw logcat result: {logcat_result[:500]}...") # Log first 500 chars

                if not logcat_result:
                    if retry == max_retries - 1:
                        logger.warning("⚠️ No logcat data received after retries")
                        return await self._get_ui_state_via_contentprovider(device, start_time)
                    continue

                # Parse UI data from logcat
                lines = logcat_result.split('\n')
                ui_data_lines = []
                capturing = False
                data_complete = False

                for line in lines:
                    if "JEEVES_UI_DATA_START" in line:
                        capturing = True
                        continue
                    elif "JEEVES_UI_DATA_END" in line:
                        capturing = False
                        data_complete = True
                        break
                    elif capturing:
                        json_str = line.strip()
                        if json_str and json_str.startswith('{') and json_str.endswith('}'):
                            ui_data_lines.append(json_str)

                # If we found complete data, break out of retry loop
                if data_complete and ui_data_lines:
                    logger.debug(f"✅ UI data found on retry {retry + 1}")
                    break

                # If we found START but not END, data is incomplete - retry
                if capturing and not data_complete:
                    logger.debug(f"⏳ Incomplete data on retry {retry + 1}, waiting...")
                    ui_data_lines = []  # Clear incomplete data
                    continue

            if not ui_data_lines:
                logger.warning("⚠️ No UI data found in logcat after retries")
                return await self._get_ui_state_via_contentprovider(device, start_time)

            # Parse JSON
            full_json_str = ''.join(ui_data_lines)
            logger.debug(f"Full JSON string from logcat: {full_json_str[:500]}...") # Log first 500 chars
            ui_data = json.loads(full_json_str)
            logger.debug(f"Parsed UI data: {ui_data.get('elements', [])[:5]}...") # Log first 5 elements

            # Convert to expected format
            elements = []
            for el in ui_data.get("elements", []):
                element = {
                    "index": el.get("idx", 0),
                    "class": el.get("cls", ""),
                    "text": el.get("txt", ""),
                    "content_desc": el.get("desc", ""),
                    "resource_id": el.get("res", ""),
                    "clickable": el.get("clk", False),
                    "enabled": el.get("en", False),
                }

                # Parse bounds
                if "bnd" in el and len(el["bnd"]) == 4:
                    element["bounds"] = {
                        "left": el["bnd"][0],
                        "top": el["bnd"][1],
                        "right": el["bnd"][2],
                        "bottom": el["bnd"][3]
                    }

                elements.append(element)

            elapsed = time.time() - start_time
            logger.info(f"🚀 Jeeves logcat broadcast: {len(elements)} elements in {elapsed:.3f}s")

            return {
                "a11y_tree": elements,
                "phone_state": {"activity": "unknown"},
                "source": "jeeves_logcat_broadcast",
                "method": "direct_logcat_transfer",
                "file_io": False,
                "element_count": len(elements),
                "timestamp": ui_data.get("ts", 0)
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in broadcast data: {e}")
            return await self._get_ui_state_via_contentprovider(device, start_time)
        except Exception as e:
            logger.error(f"Broadcast method failed: {e}")
            return await self._get_ui_state_via_contentprovider(device, start_time)

    async def _get_ui_state_via_contentprovider(self, device, start_time: float) -> Dict[str, Any]:
        """
        Get UI state via ContentProvider query - slower but reliable fallback
        """
        try:
            # Request UI state capture
            await device._adb.shell(
                device._serial,
                "am broadcast -a com.jeeves.GET_UI_STATE"
            )

            # Small delay to allow processing
            await asyncio.sleep(0.05)

            # Query ContentProvider directly
            result = await device._adb.shell(
                device._serial,
                f"content query --uri {self.content_provider_uri}"
            )

            if result and result.strip():
                ui_state = self._parse_content_provider_response(result.strip())
                elapsed = time.time() - start_time

                logger.info(f"🎩 Jeeves ContentProvider: {len(ui_state.get('a11y_tree', []))} elements in {elapsed:.3f}s")

                return ui_state
            else:
                logger.error("❌ No response from Jeeves ContentProvider")
                return {
                    "error": "Jeeves ContentProvider Empty Response",
                    "message": "ContentProvider returned no data",
                    "a11y_tree": [],
                    "phone_state": {"activity": "unknown"}
                }

        except Exception as e:
            logger.error(f"ContentProvider method failed: {e}")
            return {
                "error": "Jeeves ContentProvider Error",
                "message": str(e),
                "a11y_tree": [],
                "phone_state": {"activity": "unknown"}
            }

    async def get_ui_state_fast(self, device_serial: Optional[str] = None, show_bounds: bool = False, use_broadcast: bool = True) -> Dict[str, Any]:
        """
        Fast UI state retrieval using Jeeves Accessibility Service
        
        Args:
            device_serial: Optional device serial
            show_bounds: Show bounding boxes for debugging
            use_broadcast: Use broadcast response (fastest) vs ContentProvider
        
        Returns:
            Dictionary with 'a11y_tree' containing UI elements, similar to uiautomator format
        """
        try:
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()

            # Show bounding boxes if requested (great for debugging!)
            if show_bounds:
                await device._adb.shell(
                    device._serial,
                    "am broadcast -a com.jeeves.SHOW_BOUNDS"
                )

            start_time = time.time()

            if use_broadcast:
                # NEW: Use broadcast response approach (fastest)
                return await self._get_ui_state_via_broadcast(device, start_time)
            else:
                # FALLBACK: Use ContentProvider approach (slower but more reliable)
                return await self._get_ui_state_via_contentprovider(device, start_time)

        except Exception as e:
            logger.error(f"Error getting UI state from Jeeves: {e}")
            return {
                "error": "Jeeves Error",
                "message": str(e),
                "a11y_tree": [],
                "phone_state": {"activity": "unknown"}
            }

    async def tap_by_index_fast(self, index: int, device_serial: Optional[str] = None) -> bool:
        """
        Fast element tap using Jeeves Accessibility Service
        
        Uses direct accessibility actions when possible, falls back to gesture coordinates
        """
        try:
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()

            # Send tap command to Jeeves
            await device._adb.shell(
                device._serial,
                f"am broadcast -a com.jeeves.TAP_BY_INDEX --ei index {index}"
            )

            logger.info(f"🎯 Jeeves tap by index {index} sent")
            return True

        except Exception as e:
            logger.error(f"Error tapping by index via Jeeves: {e}")
            return False

    async def flash_touch(self, x: int, y: int, execute: bool = True, device_serial: Optional[str] = None):
        """Flash touch at coordinates (existing TouchFlasher functionality)"""
        try:
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()

            await device._adb.shell(
                device._serial,
                f"am broadcast -a com.jeeves.FLASH_TOUCH --ei x {x} --ei y {y} --ez execute {str(execute).lower()}"
            )

            logger.debug(f"✨ Flash touch at ({x}, {y}), execute={execute}")

        except Exception as e:
            logger.error(f"Error flashing touch via Jeeves: {e}")

    async def show_bounding_boxes(self, show: bool = True, device_serial: Optional[str] = None):
        """Show/hide bounding boxes overlay for debugging"""
        try:
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()

            action = "SHOW_BOUNDS" if show else "HIDE_BOUNDS"
            await device._adb.shell(
                device._serial,
                f"am broadcast -a com.jeeves.{action}"
            )

            logger.info(f"🔍 Bounding boxes {'shown' if show else 'hidden'}")

        except Exception as e:
            logger.error(f"Error controlling bounding boxes: {e}")

    async def take_screenshot_with_annotations(self, device_serial: Optional[str] = None) -> Tuple[str, bytes]:
        """Take screenshot with optional UI element annotations"""
        try:
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()

            # Show bounds briefly
            await self.show_bounding_boxes(True, device_serial)
            await asyncio.sleep(0.3)  # Let overlay render

            # Take screenshot
            screenshot_result = await device.screencap()

            # Hide bounds
            await self.show_bounding_boxes(False, device_serial)

            return screenshot_result

        except Exception as e:
            logger.error(f"Error taking annotated screenshot: {e}")
            # Fallback to regular screenshot
            device = await self.device_manager.get_device(device_serial) if device_serial else await self.device_manager.get_device()
            return await device.screencap()


class JeevesEnhancedAdbTools:
    """
    Enhanced ADB Tools that prefer Jeeves when available, fall back to uiautomator
    
    This provides a seamless upgrade path with automatic fallback
    """

    def __init__(self, original_adb_tools, device_manager):
        self.original_tools = original_adb_tools
        self.jeeves = JeevesIntegration(device_manager)
        self._jeeves_checked = False

    async def get_state_direct(self, ensure_stable: bool = False, serial: Optional[str] = None, show_bounds: bool = False) -> Dict[str, Any]:
        """
        Get UI state with automatic Jeeves/uiautomator selection
        """
        # Check Jeeves availability once
        if not self._jeeves_checked:
            jeeves_available = await self.jeeves.is_jeeves_available(serial)
            self._jeeves_checked = True

            if jeeves_available:
                logger.info("🎩 Using Jeeves for fast UI state access")
            else:
                logger.info("🐌 Falling back to uiautomator (Jeeves unavailable)")

        # Try Jeeves first if available
        if self.jeeves.is_available:
            try:
                return await self.jeeves.get_ui_state_fast(serial, show_bounds)
            except Exception as e:
                logger.warning(f"Jeeves failed, falling back to uiautomator: {e}")

        # Fallback to original uiautomator method
        return await self.original_tools.get_state_direct(ensure_stable, serial)

    async def tap_by_index(self, index: int, serial: Optional[str] = None) -> str:
        """Tap by index with Jeeves preference"""
        if self.jeeves.is_available:
            try:
                success = await self.jeeves.tap_by_index_fast(index, serial)
                if success:
                    return f"Tapped element with index {index} via Jeeves"
            except Exception as e:
                logger.warning(f"Jeeves tap failed, falling back: {e}")

        # Fallback to original method
        return await self.original_tools.tap_by_index(index, serial)

    async def take_screenshot(self, serial: Optional[str] = None) -> Tuple[str, bytes]:
        """Take screenshot with optional Jeeves annotations"""
        if self.jeeves.is_available:
            try:
                return await self.jeeves.take_screenshot_with_annotations(serial)
            except Exception as e:
                logger.debug(f"Jeeves screenshot failed, using standard: {e}")

        # Fallback to original method
        return await self.original_tools.take_screenshot(serial)

    def __getattr__(self, name):
        """Delegate all other methods to original AdbTools"""
        return getattr(self.original_tools, name)


def enhance_adb_tools_with_jeeves(original_adb_tools, device_manager):
    """
    Factory function to enhance existing AdbTools with Jeeves capabilities
    
    Usage:
        enhanced_tools = enhance_adb_tools_with_jeeves(adb_tools, device_manager)
        # Now enhanced_tools will use Jeeves when available, uiautomator as fallback
    """
    return JeevesEnhancedAdbTools(original_adb_tools, device_manager)


# Testing and validation functions
async def test_jeeves_integration():
    """Test script to validate Jeeves integration"""
    from droidrun.tools.adb import AdbTools
    from droidrun.tools.device_manager import DeviceManager

    print("🧪 Testing Jeeves Integration")
    print("=" * 50)

    # Setup
    device_manager = DeviceManager()
    await device_manager.setup_device_connection()

    original_tools = AdbTools()
    original_tools.device_manager = device_manager

    enhanced_tools = enhance_adb_tools_with_jeeves(original_tools, device_manager)

    # Test 1: Availability check
    print("\n1. Checking Jeeves availability...")
    available = await enhanced_tools.jeeves.is_jeeves_available()
    print(f"   Jeeves available: {available}")

    if available:
        # Test 2: UI state comparison
        print("\n2. Comparing UI state performance...")

        # Jeeves method
        start = time.time()
        jeeves_state = await enhanced_tools.jeeves.get_ui_state_fast(show_bounds=True)
        jeeves_time = time.time() - start
        jeeves_elements = len(jeeves_state.get('a11y_tree', []))

        # Hide bounds
        await enhanced_tools.jeeves.show_bounding_boxes(False)

        # uiautomator method
        start = time.time()
        uiautomator_state = await original_tools.get_state_direct()
        uiautomator_time = time.time() - start
        uiautomator_elements = len(uiautomator_state.get('a11y_tree', []))

        print(f"   Jeeves:      {jeeves_elements} elements in {jeeves_time:.3f}s")
        print(f"   UIAutomator: {uiautomator_elements} elements in {uiautomator_time:.3f}s")
        print(f"   Speedup:     {uiautomator_time / jeeves_time:.1f}x faster")

        # Test 3: Tap functionality
        print("\n3. Testing tap functionality...")
        if jeeves_elements > 0:
            print("   Showing bounds and testing tap...")
            await enhanced_tools.jeeves.show_bounding_boxes(True)
            await asyncio.sleep(2)  # Let user see
            await enhanced_tools.jeeves.show_bounding_boxes(False)
    else:
        print("\n⚠️  Jeeves not available. Install and enable accessibility service.")
        print("   Installation:")
        print("   1. adb install Jeeves.apk")
        print("   2. Enable accessibility service in Settings > Accessibility")

    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_jeeves_integration())
