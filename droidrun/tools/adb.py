"""
UI Actions - Core UI interaction tools for Android device control.
"""

import signal
import shlex
import asyncio
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import io

from droidrun.adb.device import Device
from droidrun.adb.manager import DeviceManager
from droidrun.tools.tools import Tools

logger = logging.getLogger("droidrun-adb-tools")


class AdbTools(Tools):
    """Core UI interaction tools for Android device control."""

    # Class-level lock to prevent concurrent uiautomator calls across all instances
    _uiautomator_lock = asyncio.Lock()

    # Global toggle for UI stability checking (set via environment variable or directly)
    UI_STABILITY_CHECK = os.environ.get("UI_STABILITY_CHECK", "auto").lower()

    @classmethod
    def set_stability_check(cls, mode: str):
        """Set global UI stability checking mode.

        Args:
            mode: 'true'/'1' (always on), 'false'/'0' (always off), or 'auto' (smart mode)
        """
        cls.UI_STABILITY_CHECK = str(mode).lower()
        logger.info(f"🔧 UI stability checking set to: {cls.UI_STABILITY_CHECK}")

    def __init__(
        self,
        serial: str,
        host_volume_path: str,
        container_mount_path: str,
        adb_path: str = "adb",
        ensure_ui_stability: Optional[bool] = None,
    ) -> None:
        # Instance‐level cache for clickable elements (index-based tapping)
        self.clickable_elements_cache: List[Dict[str, Any]] = []

        # Instance-level UI stability setting (overrides global if set)
        self.ensure_ui_stability = ensure_ui_stability
        self.serial = serial
        self.adb_path = adb_path
        self.device_manager = DeviceManager(adb_path=adb_path)
        self.last_screenshot = None
        self.reason = None
        self.success = None
        self.finished = False
        # Memory storage for remembering important information
        self.memory: List[str] = []
        # Store all screenshots with timestamps
        self.screenshots: List[Dict[str, Any]] = []
        # Screenshot counter for naming
        self.screenshot_counter = 0
        # Directory for saving screenshots
        self.screenshot_save_dir: Optional[str] = None

        self.recording_process: Optional[asyncio.subprocess.Process] = None
        self.recording_host_path: Optional[str] = None
        self.host_volume_path = os.path.abspath(host_volume_path)  # Resolve to an absolute path
        self.container_mount_path = container_mount_path

    def set_screenshot_save_dir(self, directory: str) -> None:
        """Set the directory where screenshots should be saved.

        Args:
            directory: Path to directory where screenshots will be saved
        """
        self.screenshot_save_dir = directory
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Screenshot save directory set to: {directory}")

    def should_ensure_stability(self, for_screenshot: bool = False) -> bool:
        """Determine if UI stability checking should be used.

        Priority order:
        1. Instance-level setting (if explicitly set)
        2. Environment variable UI_STABILITY_CHECK
        3. Default behavior based on context

        Args:
            for_screenshot: If True, checking for screenshot annotation context

        Returns:
            bool: Whether to ensure UI stability
        """
        # If instance has explicit setting, use it
        if self.ensure_ui_stability is not None:
            return self.ensure_ui_stability

        # Check global setting
        global_setting = self.UI_STABILITY_CHECK

        if global_setting == "true" or global_setting == "1":
            return True
        elif global_setting == "false" or global_setting == "0":
            return False
        elif global_setting == "auto":
            # Auto mode: enable for screenshots (training data), disable otherwise
            return for_screenshot
        else:
            # Default to auto behavior
            return for_screenshot

    def get_device_serial(self) -> str:
        """Get the device serial from the instance or environment variable."""
        # First try using the instance's serial
        if self.serial:
            return self.serial

    async def get_device(self) -> Optional[Device]:
        """Get the device instance using the instance's serial or from environment variable.

        Returns:
            Device instance or None if not found
        """
        serial = self.get_device_serial()
        if not serial:
            raise ValueError("No device serial specified - set device_serial parameter")

        device = await self.device_manager.get_device(serial)
        if not device:
            raise ValueError(f"Device {serial} not found")

        return device

    def parse_package_list(self, output: str) -> List[Dict[str, str]]:
        """Parse the output of 'pm list packages -f' command.

        Args:
            output: Raw command output from 'pm list packages -f'

        Returns:
            List of dictionaries containing package info with 'package' and 'path' keys
        """
        apps = []
        for line in output.splitlines():
            if line.startswith("package:"):
                # Format is: "package:/path/to/base.apk=com.package.name"
                path_and_pkg = line[8:]  # Strip "package:"
                if "=" in path_and_pkg:
                    path, package = path_and_pkg.rsplit("=", 1)
                    apps.append({"package": package.strip(), "path": path.strip()})
        return apps

    def _parse_content_provider_output(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """
        Parse the raw ADB content provider output and extract JSON data.

        Args:
            raw_output (str): Raw output from ADB content query command

        Returns:
            dict: Parsed JSON data or None if parsing failed
        """
        # The ADB content query output format is: "Row: 0 result={json_data}"
        # We need to extract the JSON part after "result="
        lines = raw_output.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Look for lines that contain "result=" pattern
            if "result=" in line:
                # Extract everything after "result="
                result_start = line.find("result=") + 7
                json_str = line[result_start:]

                try:
                    # Parse the JSON string
                    json_data = json.loads(json_str)
                    return json_data
                except json.JSONDecodeError:
                    continue

            # Fallback: try to parse lines that start with { or [
            elif line.startswith("{") or line.startswith("["):
                try:
                    json_data = json.loads(line)
                    return json_data
                except json.JSONDecodeError:
                    continue

        # If no valid JSON found in individual lines, try the entire output
        try:
            json_data = json.loads(raw_output.strip())
            return json_data
        except json.JSONDecodeError:
            return None

    async def tap_by_index(self, index: int, serial: Optional[str] = None) -> str:
        """
        Tap on a UI element by its index.

        This function uses the cached clickable elements
        to find the element with the given index and tap on its center coordinates.

        Args:
            index: Index of the element to tap

        Returns:
            Result message
        """

        def collect_all_indices(elements):
            """Recursively collect all indices from elements and their children."""
            indices = []
            for item in elements:
                if item.get("index") is not None:
                    indices.append(item.get("index"))
                # Check children if present
                children = item.get("children", [])
                indices.extend(collect_all_indices(children))
            return indices

        def find_element_by_index(elements, target_index):
            """Recursively find an element with the given index."""
            for item in elements:
                if item.get("index") == target_index:
                    return item
                # Check children if present
                children = item.get("children", [])
                result = find_element_by_index(children, target_index)
                if result:
                    return result
            return None

        try:
            # Check if we have cached elements
            if not self.clickable_elements_cache:
                return "Error: No UI elements cached. Call get_state first."

            # Find the element with the given index (including in children)
            element = find_element_by_index(self.clickable_elements_cache, index)

            if not element:
                # List available indices to help the user
                indices = sorted(collect_all_indices(self.clickable_elements_cache))
                indices_str = ", ".join(str(idx) for idx in indices[:20])
                if len(indices) > 20:
                    indices_str += f"... and {len(indices) - 20} more"

                return (
                    f"Error: No element found with index {index}. Available indices: {indices_str}"
                )

            # Get the bounds of the element
            bounds_str = element.get("bounds")
            if not bounds_str:
                element_text = element.get("text", "No text")
                element_type = element.get("type", "unknown")
                element_class = element.get("className", "Unknown class")
                return f"Error: Element with index {index} ('{element_text}', {element_class}, type: {element_type}) has no bounds and cannot be tapped"

            # Parse the bounds (format: "left,top,right,bottom")
            try:
                left, top, right, bottom = map(int, bounds_str.split(","))
            except ValueError:
                return f"Error: Invalid bounds format for element with index {index}: {bounds_str}"

            # Calculate the center of the element
            x = (left + right) // 2
            y = (top + bottom) // 2

            # Get the device and tap at the coordinates
            if serial:
                device = await self.device_manager.get_device(serial)
                if not device:
                    return f"Error: Device {serial} not found"
            else:
                device = await self.get_device()

            await device.tap(x, y)

            # Add a small delay to allow UI to update
            await asyncio.sleep(0.5)

            # Create a descriptive response
            response_parts = []
            response_parts.append(f"Tapped element with index {index}")
            response_parts.append(f"Text: '{element.get('text', 'No text')}'")
            response_parts.append(f"Class: {element.get('className', 'Unknown class')}")
            response_parts.append(f"Type: {element.get('type', 'unknown')}")

            # Add information about children if present
            children = element.get("children", [])
            if children:
                child_texts = [child.get("text") for child in children if child.get("text")]
                if child_texts:
                    response_parts.append(f"Contains text: {' | '.join(child_texts)}")

            response_parts.append(f"Coordinates: ({x}, {y})")

            result = " | ".join(response_parts)
            print(result)  # Print so SimpleCodeExecutor can capture it
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    # Rename the old tap function to tap_by_coordinates for backward compatibility
    async def tap_by_coordinates(self, x: int, y: int) -> bool:
        """
        Tap on the device screen at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Bool indicating success or failure
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()

            await device.tap(x, y)
            print(f"Tapped at coordinates ({x}, {y})")
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    # Replace the old tap function with the new one
    async def tap(self, index: int, wait_after: float = 0.3) -> str:
        """
        Tap on a UI element by its index.

        This function uses the cached clickable elements from the last get_clickables call
        to find the element with the given index and tap on its center coordinates.

        Args:
            index: Index of the element to tap
            wait_after: Time to wait after tap (seconds) for UI to stabilize

        Returns:
            Result message
        """
        result = await self.tap_by_index(index)
        if wait_after > 0:
            await asyncio.sleep(wait_after)
        return result

    async def swipe(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300
    ) -> bool:
        """
        Performs a straight-line swipe gesture on the device screen.
        To perform a hold (long press), set the start and end coordinates to the same values and increase the duration as needed.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds
        Returns:
            Bool indicating success or failure
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()

            await device.swipe(start_x, start_y, end_x, end_y, duration_ms)
            await asyncio.sleep(1)
            print(f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms}ms")
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    async def input_text(self, text: str, serial: Optional[str] = None) -> str:
        """
        Input text on the device.
        Always make sure that the Focused Element is not None before inputting text.

        Args:
            text: Text to input. Can contain spaces, newlines, and special characters including non-ASCII.

        Returns:
            Result message
        """
        try:
            if serial:
                device = await self.device_manager.get_device(serial)
                if not device:
                    return f"Error: Device {serial} not found"
            else:
                device = await self.get_device()

            # Save the current keyboard
            original_ime = await device._adb.shell(
                device._serial, "settings get secure default_input_method"
            )
            original_ime = original_ime.strip()

            # Enable the Droidrun keyboard
            await device._adb.shell(
                device._serial, "ime enable com.droidrun.portal/.DroidrunKeyboardIME"
            )

            # Set the Droidrun keyboard as the default
            await device._adb.shell(
                device._serial, "ime set com.droidrun.portal/.DroidrunKeyboardIME"
            )

            # Wait for keyboard to change
            await asyncio.sleep(1)

            # Encode the text to Base64
            import base64

            encoded_text = base64.b64encode(text.encode()).decode()

            cmd = f'content insert --uri "content://com.droidrun.portal/keyboard/input" --bind base64_text:s:"{encoded_text}"'
            await device._adb.shell(device._serial, cmd)

            # Wait for text input to complete
            await asyncio.sleep(0.5)

            # Restore the original keyboard
            if original_ime and "com.droidrun.portal" not in original_ime:
                await device._adb.shell(device._serial, f"ime set {original_ime}")

            result = f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"
            print(result)  # Print so SimpleCodeExecutor can capture it
            return result
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error sending text input: {str(e)}"

    async def back(self) -> str:
        """
        Go back on the current view.
        This presses the Android back button.
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()

            await device.press_key(3)
            result = "Pressed key BACK"
            print(result)  # Print so SimpleCodeExecutor can capture it
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def press_key(self, keycode: int) -> str:
        """
        Press a key on the Android device.

        Common keycodes:
        - 3: HOME
        - 4: BACK
        - 66: ENTER
        - 67: DELETE

        Args:
            keycode: Android keycode to press
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()

            key_names = {
                66: "ENTER",
                4: "BACK",
                3: "HOME",
                67: "DELETE",
            }
            key_name = key_names.get(keycode, str(keycode))

            await device.press_key(keycode)
            result = f"Pressed key {key_name}"
            print(result)  # Print so SimpleCodeExecutor can capture it
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def start_app(self, package: str, activity: str = "") -> str:
        """
        Start an app on the device.

        Args:
            package: Package name (e.g., "com.android.settings")
            activity: Optional activity name
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()

            result = await device.start_app(package, activity)
            print(result)  # Print so SimpleCodeExecutor can capture it
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def install_app(
        self, apk_path: str, reinstall: bool = False, grant_permissions: bool = True
    ) -> str:
        """
        Install an app on the device.

        Args:
            apk_path: Path to the APK file
            reinstall: Whether to reinstall if app exists
            grant_permissions: Whether to grant all permissions
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()

            if not os.path.exists(apk_path):
                return f"Error: APK file not found at {apk_path}"

            result = await device.install_app(apk_path, reinstall, grant_permissions)
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def take_screenshot(self) -> Tuple[str, bytes]:
        """
        Take a screenshot of the device.
        This function captures the current screen and adds the screenshot to context in the next message.
        Also stores the screenshot in the screenshots list with timestamp for later GIF creation.
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    raise ValueError(f"Device {self.serial} not found")
            else:
                device = await self.get_device()
            # Use high quality (95) for vision models to preserve detail
            screen_tuple = await device.take_screenshot(quality=95)
            self.last_screenshot = screen_tuple[1]

            # Store screenshot with timestamp
            timestamp = time.time()
            self.screenshots.append(
                {
                    "timestamp": timestamp,
                    "image_data": screen_tuple[1],
                    "format": screen_tuple[0],  # Usually 'PNG'
                }
            )

            return screen_tuple
        except ValueError as e:
            raise ValueError(f"Error taking screenshot: {str(e)}")

    def create_annotated_screenshot(
        self, screenshot_bytes: bytes, clickable_elements: List[Dict[str, Any]]
    ) -> bytes:
        """
        Create a screenshot with numbered bounding boxes overlaid on clickable elements.

        Args:
            screenshot_bytes: Raw PNG screenshot bytes
            clickable_elements: List of clickable elements with bounds and indices

        Returns:
            bytes: Annotated screenshot as PNG bytes
        """
        try:
            # Open the screenshot image
            img = Image.open(io.BytesIO(screenshot_bytes))
            draw = ImageDraw.Draw(img)

            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
                )
            except (OSError, IOError):
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

            def process_elements(elements, draw, font):
                """Recursively process elements and their children."""
                for element in elements:
                    # Skip elements without bounds or index
                    bounds_str = element.get("bounds")
                    index = element.get("index")

                    if bounds_str and index is not None:
                        try:
                            # Parse bounds: "left,top,right,bottom"
                            coords = bounds_str.split(",")
                            if len(coords) == 4:
                                left, top, right, bottom = map(int, coords)

                                STATUS_BAR_HEIGHT = (
                                    0  # Test with no offset first to check alignment
                                )
                                top += STATUS_BAR_HEIGHT
                                bottom += STATUS_BAR_HEIGHT

                                # Draw red rectangle for bounding box
                                colours = ["green", "blue", "red", "purple"]
                                colour = colours[index % len(colours)]
                                draw.rectangle(
                                    [(left, top), (right, bottom)],
                                    outline=colour,
                                    width=3,
                                )

                                # Draw white circle with red border for number background
                                circle_center_x = left + 30
                                circle_center_y = top + 30
                                circle_radius = 30

                                draw.ellipse(
                                    [
                                        (
                                            circle_center_x - circle_radius,
                                            circle_center_y - circle_radius,
                                        ),
                                        (
                                            circle_center_x + circle_radius,
                                            circle_center_y + circle_radius,
                                        ),
                                    ],
                                    fill="white",
                                    outline=colour,
                                    width=2,
                                )

                                # Draw the index number
                                number_text = str(index)
                                if font:
                                    # Get text bounding box to center it
                                    bbox = draw.textbbox((0, 0), number_text, font=font)
                                    text_width = bbox[2] - bbox[0]
                                    text_height = bbox[3] - bbox[1]

                                    text_x = circle_center_x - text_width // 2
                                    text_y = circle_center_y - text_height // 2

                                    draw.text((text_x, text_y), number_text, fill=colour, font=font)
                                else:
                                    # Fallback without font
                                    draw.text(
                                        (circle_center_x - 5, circle_center_y - 8),
                                        number_text,
                                        fill=colour,
                                    )

                        except (ValueError, IndexError) as e:
                            logger.debug(
                                f"Could not parse bounds '{bounds_str}' for element {index}: {e}"
                            )

                    # Process children recursively
                    children = element.get("children", [])
                    if children:
                        process_elements(children, draw, font)

            # Process all elements
            process_elements(clickable_elements, draw, font)

            # Convert back to bytes
            output = io.BytesIO()
            img.save(output, format="PNG")
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Failed to create annotated screenshot: {e}")
            # Return original screenshot if annotation fails
            return screenshot_bytes

    async def take_annotated_screenshot(self) -> Tuple[str, bytes]:
        """
        Take a screenshot with numbered bounding boxes overlaid on clickable elements.

        Returns:
            Tuple[str, bytes]: Format and annotated screenshot bytes
        """
        logger.info(f"🎯 take_annotated_screenshot CALLED")
        try:
            # Take regular screenshot first
            screenshot_result = await self.take_screenshot()
            screenshot_bytes = screenshot_result[1]

            logger.info("🔍 Getting fresh UI state for annotation...")
            # Check if stability checking should be used for screenshots
            use_stability = self.should_ensure_stability(for_screenshot=True)
            if use_stability:
                logger.info("📊 UI stability checking is ENABLED for screenshot")
            fresh_state = await self.get_state_direct(ensure_stable=use_stability)
            current_elements = fresh_state.get("a11y_tree", [])

            if not current_elements:
                logger.error(
                    f"No clickable elements found! fresh_state keys: {list(fresh_state.keys())}, "
                    f"fresh_state content: {fresh_state}"
                )
                raise Exception("No clickable elements found for annotation")

            # Create annotated version using FRESH elements from current screen
            # CRITICAL: Use fresh elements, not stale cache
            elements_snapshot = [
                element.copy() for element in current_elements if isinstance(element, dict)
            ]
            annotated_bytes = self.create_annotated_screenshot(screenshot_bytes, elements_snapshot)

            # CRITICAL: Save the EXACT annotated screenshot that will be sent to the LLM
            if self.screenshot_save_dir:
                self.screenshot_counter += 1
                from datetime import datetime

                timestamp = time.time()
                dt = datetime.fromtimestamp(timestamp)
                # Save as LLM screenshot - this is the EXACT bytes shown to LLM
                filename = f"{self.screenshot_counter:04d}_{dt.strftime('%Y%m%d_%H%M%S_%f')}_llm_screenshot.png"
                filepath = os.path.join(self.screenshot_save_dir, filename)

                try:
                    with open(filepath, "wb") as f:
                        f.write(annotated_bytes)
                    logger.info(f"🎯 EXACT LLM screenshot saved to: {filepath}")

                    # ALSO save the elements snapshot used for this screenshot
                    elements_file = filepath.replace("_llm_screenshot.png", "_elements.json")
                    with open(elements_file, "w") as f:
                        json.dump(elements_snapshot, f, indent=2)
                    logger.info(f"🎯 Elements snapshot saved to: {elements_file}")

                except Exception as e:
                    logger.error(f"CRITICAL: Failed to save LLM screenshot to {filepath}: {e}")
            else:
                logger.error(
                    f"CRITICAL: No screenshot save directory set - cannot save LLM screenshot!"
                )

            return (screenshot_result[0], annotated_bytes)

        except Exception as e:
            logger.error(f"Error taking annotated screenshot: {e}")
            # No fallback - raise the error instead of taking plain screenshots
            raise e

    async def list_packages(self, include_system_apps: bool = False) -> List[str]:
        """
        List installed packages on the device.

        Args:
            include_system_apps: Whether to include system apps (default: False)

        Returns:
            List of package names
        """
        try:
            if self.serial:
                device = await self.device_manager.get_device(self.serial)
                if not device:
                    raise ValueError(f"Device {self.serial} not found")
            else:
                device = await self.get_device()

            # Use the direct ADB command to get packages with paths
            cmd = ["pm", "list", "packages", "-f"]
            if not include_system_apps:
                cmd.append("-3")

            output = await device._adb.shell(device._serial, " ".join(cmd))

            # Parse the package list using the function
            packages = self.parse_package_list(output)
            # Format package list for better readability
            package_list = [pack["package"] for pack in packages]
            for package in package_list:
                print(package)
            return package_list
        except ValueError as e:
            raise ValueError(f"Error listing packages: {str(e)}")

    async def extract(self, filename: Optional[str] = None) -> str:
        """Extract and save the current UI state to a JSON file.

        This function captures the current UI state including all UI elements
        and saves it to a JSON file for later analysis or reference.

        Args:
            filename: Optional filename to save the UI state (defaults to ui_state_TIMESTAMP.json)

        Returns:
            Path to the saved JSON file
        """
        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"ui_state_{timestamp}.json"

            # Ensure the filename ends with .json
            if not filename.endswith(".json"):
                filename += ".json"

            # Get the UI elements
            ui_elements = await self.get_all_elements(self.serial)

            # Save to file
            save_path = os.path.abspath(filename)
            async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(ui_elements, indent=2))

            return f"UI state extracted and saved to {save_path}"

        except Exception as e:
            return f"Error extracting UI state: {e}"

    async def get_all_elements(self) -> Dict[str, Any]:
        """
        Get all UI elements from the device, including non-interactive elements.

        This function interacts with the TopViewService app installed on the device
        to capture all UI elements, even those that are not interactive. This provides
        a complete view of the UI hierarchy for analysis or debugging purposes.

        Returns:
            Dictionary containing all UI elements extracted from the device screen
        """
        try:
            # Get the device
            device = await self.device_manager.get_device(self.serial)
            if not device:
                raise ValueError(f"Device {self.serial} not found")

            # Create a temporary file for the JSON
            with tempfile.NamedTemporaryFile(suffix=".json") as temp:
                local_path = temp.name

                try:
                    # Clear logcat to make it easier to find our output
                    await device._adb.shell(device._serial, "logcat -c")

                    # Trigger the custom service via broadcast to get ALL elements
                    await device._adb.shell(
                        device._serial,
                        "am broadcast -a com.droidrun.portal.GET_ALL_ELEMENTS",
                    )

                    # Poll for the JSON file path
                    start_time = asyncio.get_event_loop().time()
                    max_wait_time = 10  # Maximum wait time in seconds
                    poll_interval = 0.2  # Check every 200ms

                    device_path = None
                    while asyncio.get_event_loop().time() - start_time < max_wait_time:
                        # Check logcat for the file path
                        logcat_output = await device._adb.shell(
                            device._serial,
                            'logcat -d | grep "DROIDRUN_FILE" | grep "JSON data written to" | tail -1',
                        )

                        # Parse the file path if present
                        match = re.search(r"JSON data written to: (.*)", logcat_output)
                        if match:
                            device_path = match.group(1).strip()
                            break

                        # Wait before polling again
                        await asyncio.sleep(poll_interval)

                    # Check if we found the file path
                    if not device_path:
                        raise ValueError(
                            f"Failed to find the JSON file path in logcat after {max_wait_time} seconds"
                        )

                    logger.debug(f"Pulling file from {device_path} to {local_path}")
                    # Pull the JSON file from the device
                    await device._adb.pull_file(device._serial, device_path, local_path)

                    # Read the JSON file
                    async with aiofiles.open(local_path, "r", encoding="utf-8") as f:
                        json_content = await f.read()

                    # Clean up the temporary file
                    with contextlib.suppress(OSError):
                        os.unlink(local_path)

                    # Try to parse the JSON
                    import json

                    try:
                        ui_data = json.loads(json_content)

                        return {
                            "all_elements": ui_data,
                            "count": (
                                len(ui_data)
                                if isinstance(ui_data, list)
                                else sum(1 for _ in ui_data.get("elements", []))
                            ),
                            "message": "Retrieved all UI elements from the device screen",
                        }
                    except json.JSONDecodeError:
                        raise ValueError("Failed to parse UI elements JSON data")

                except Exception as e:
                    # Clean up in case of error
                    with contextlib.suppress(OSError):
                        os.unlink(local_path)
                    raise ValueError(f"Error retrieving all UI elements: {e}")

        except Exception as e:
            raise ValueError(f"Error getting all UI elements: {e}")

    def complete(self, success: bool, reason: str = ""):
        """
        Mark the task as finished.

        Args:
            success: Indicates if the task was successful.
            reason: Reason for failure/success
        """
        if success:
            self.success = True
            self.reason = reason or "Task completed successfully."
            self.finished = True
        else:
            self.success = False
            if not reason:
                raise ValueError("Reason for failure is required if success is False.")
            self.reason = reason
            self.finished = True

    async def remember(self, information: str) -> str:
        """
        Store important information to remember for future context.

        This information will be extracted and included into your next steps to maintain context
        across interactions. Use this for critical facts, observations, or user preferences
        that should influence future decisions.

        Args:
            information: The information to remember

        Returns:
            Confirmation message
        """
        if not information or not isinstance(information, str):
            return "Error: Please provide valid information to remember."

        # Add the information to memory
        self.memory.append(information.strip())

        # Limit memory size to prevent context overflow (keep most recent items)
        max_memory_items = 10
        if len(self.memory) > max_memory_items:
            self.memory = self.memory[-max_memory_items:]

        return f"Remembered: {information}"

    def get_memory(self) -> List[str]:
        """
        Retrieve all stored memory items.

        Returns:
            List of stored memory items
        """
        return self.memory.copy()

    async def get_state(self, serial: Optional[str] = None) -> Dict[str, Any]:
        """
        Get both the a11y tree and phone state in a single call using the combined /state endpoint.

        Args:
            serial: Optional device serial number

        Returns:
            Dictionary containing both 'a11y_tree' and 'phone_state' data
        """

        try:
            if serial:
                device = await self.device_manager.get_device(serial)
                if not device:
                    raise ValueError(f"Device {serial} not found")
            else:
                device = await self.get_device()

            return await self.get_state_direct(serial)

        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error getting combined state: {str(e)}",
            }

    async def get_state_direct(
        self, serial: Optional[str] = None, ensure_stable: bool = False
    ) -> Dict[str, Any]:
        """
        Get fresh accessibility tree directly from Android uiautomator, bypassing Portal cache.

        Args:
            serial: Optional device serial number
            ensure_stable: If True, verify UI is stable by comparing consecutive dumps

        Returns:
            Dictionary containing fresh 'a11y_tree' data directly from Android
        """
        import xml.etree.ElementTree as ET

        try:
            if serial:
                device = await self.device_manager.get_device(serial)
                if not device:
                    raise ValueError(f"Device {serial} not found")
            else:
                device = await self.get_device()

            # Use native uiautomator to get FRESH accessibility tree directly
            logger.info("🔍 Getting fresh UI state directly from Android uiautomator...")

            # CRITICAL: Acquire lock to prevent concurrent uiautomator calls
            async with AdbTools._uiautomator_lock:
                logger.debug("🔒 Acquired uiautomator lock")

                # Retry up to 3 times with simple timeout protection
                for attempt in range(3):
                    try:
                        # Run uiautomator dump directly with timeout protection
                        adb_output = await asyncio.wait_for(
                            device._adb.shell(
                                device._serial,
                                "uiautomator dump /sdcard/ui_tree.xml && cat /sdcard/ui_tree.xml",
                            ),
                            timeout=8.0,  # 8 second timeout
                        )

                        logger.debug(
                            f"uiautomator raw output length: {len(adb_output) if adb_output else 0}"
                        )

                        # Proactive cleanup after successful call to prevent resource accumulation
                        try:
                            await device._adb.shell(device._serial, "rm -f /sdcard/ui_tree.xml")
                            await asyncio.sleep(0.1)  # Brief pause to let cleanup complete
                        except:
                            pass  # Ignore cleanup errors

                        break  # Success, exit retry loop
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning(f"uiautomator attempt {attempt + 1}/3 failed: {e}")
                        if attempt < 2:  # Not the last attempt
                            # Keep-alive: restart uiautomator service to clear stuck processes
                            try:
                                logger.info("🔄 Restarting uiautomator service as keep-alive...")
                                await device._adb.shell(device._serial, "killall uiautomator")
                                await asyncio.sleep(0.5)
                            except:
                                pass  # Ignore restart errors
                        else:  # Last attempt failed
                            return {
                                "error": "UIAutomator Failed",
                                "message": f"uiautomator command failed after 3 attempts: {str(e)}",
                            }

            if not adb_output or "UI hierchary dumped" not in adb_output:
                return {
                    "error": "Dump Error",
                    "message": "Failed to dump UI hierarchy with uiautomator",
                }

            # Extract XML content (after the dump confirmation message)
            xml_start = adb_output.find("<?xml")
            if xml_start == -1:
                return {
                    "error": "Parse Error",
                    "message": "No XML found in uiautomator output",
                }

            xml_content = adb_output[xml_start:]

            # If ensure_stable is True, verify UI has stabilized
            if ensure_stable:
                logger.info("🔄 Ensuring UI stability before returning state...")
                await asyncio.sleep(0.5)  # Brief pause

                # Get a second dump to compare
                try:
                    second_dump = await asyncio.wait_for(
                        device._adb.shell(
                            device._serial,
                            "uiautomator dump /sdcard/ui_tree2.xml && cat /sdcard/ui_tree2.xml",
                        ),
                        timeout=5.0,
                    )

                    # Clean up second dump file
                    try:
                        await device._adb.shell(device._serial, "rm -f /sdcard/ui_tree2.xml")
                    except:
                        pass

                    # Compare the two dumps
                    if second_dump and "<?xml" in second_dump:
                        second_xml_start = second_dump.find("<?xml")
                        second_xml = second_dump[second_xml_start:]

                        # If XML content differs significantly, use the newer one
                        if len(second_xml) != len(xml_content):
                            logger.info("🔄 UI changed, using newer dump")
                            xml_content = second_xml
                        else:
                            logger.debug("✅ UI appears stable")
                except:
                    # If second dump fails, continue with first
                    logger.warning("Could not verify UI stability, proceeding with initial dump")

            # Parse XML into accessibility elements
            try:
                root = ET.fromstring(xml_content)
                # Reset counter for each parse to start from 0
                elements = self._parse_ui_elements(root, global_index_counter=[0])

                # Update cache with fresh data - this maintains the mapping
                self.clickable_elements_cache = elements

                return {
                    "a11y_tree": elements,
                    "phone_state": {},  # Placeholder - could add dumpsys calls if needed
                }

            except ET.ParseError as e:
                return {
                    "error": "XML Parse Error",
                    "message": f"Failed to parse XML: {str(e)}",
                }

        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error getting direct state: {str(e)}",
            }

    def _parse_ui_elements(self, node, global_index_counter=[0]) -> List[Dict[str, Any]]:
        """Parse XML UI hierarchy into accessibility tree format with globally unique indices."""
        elements = []

        # Convert XML node to accessibility element format
        bounds_str = node.attrib.get("bounds", "[0,0][0,0]")
        # Parse bounds format: [x1,y1][x2,y2] -> (x1, y1, x2, y2)
        import re

        bounds_match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if bounds_match:
            x1, y1, x2, y2 = map(int, bounds_match.groups())
            bounds = (x1, y1, x2, y2)
        else:
            bounds = (0, 0, 0, 0)

        element = {
            "class": node.attrib.get("class", ""),
            "package": node.attrib.get("package", ""),
            "content-desc": node.attrib.get("content-desc", ""),
            "text": node.attrib.get("text", ""),
            "resource-id": node.attrib.get("resource-id", ""),
            "checkable": node.attrib.get("checkable", "false") == "true",
            "checked": node.attrib.get("checked", "false") == "true",
            "clickable": node.attrib.get("clickable", "false") == "true",
            "enabled": node.attrib.get("enabled", "true") == "true",
            "focusable": node.attrib.get("focusable", "false") == "true",
            "focused": node.attrib.get("focused", "false") == "true",
            "scrollable": node.attrib.get("scrollable", "false") == "true",
            "long-clickable": node.attrib.get("long-clickable", "false") == "true",
            "password": node.attrib.get("password", "false") == "true",
            "selected": node.attrib.get("selected", "false") == "true",
            "bounds": f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
        }

        # Only include elements that meet visibility criteria
        if (
            element["enabled"]
            and bounds[0] < bounds[2]
            and bounds[1] < bounds[3]  # Valid bounds
            and (element["clickable"] or element["text"] or element["content-desc"])
        ):
            # Assign globally unique index using counter
            element["index"] = global_index_counter[0]
            global_index_counter[0] += 1
            elements.append(element)

        # Process child nodes recursively
        for child in node:
            elements.extend(self._parse_ui_elements(child, global_index_counter))

        return elements

    # CHANGED: Updated start_recording method
    async def start_recording(
        self,
        dpath: os.PathLike,
        output_filename: str = "recording.mp4",
        bit_rate_mbps: int = 8,
        max_file_size_mb: int = 400,
    ) -> str:
        """
        Start a screen recording, saving it to the task-specific directory.

        Args:
            dpath: The directory path on the HOST for the current task's results.
            output_filename: The name of the video file.

        Returns:
            A confirmation message or an error string.
        """
        if self.recording_process:
            return "Error: A recording is already in progress."

        # --- Calculate time limit from file size and bit rate ---
        # (Size in Megabytes * 8 bits_per_byte) / Megabits_per_second = Duration in seconds
        # This prevents the recording from exceeding the specified file size.
        max_duration_seconds = math.ceil((max_file_size_mb * 8) / bit_rate_mbps)

        # --- Path Translation Logic (remains the same) ---
        abs_dpath_host = os.path.abspath(dpath)
        if not abs_dpath_host.startswith(self.host_volume_path):
            return (
                f"Error: The provided path '{dpath}' is not inside the configured host volume "
                f"path '{self.host_volume_path}'."
            )

        container_dpath = abs_dpath_host.replace(
            self.host_volume_path, self.container_mount_path, 1
        )
        container_recording_path = os.path.join(container_dpath, output_filename)
        self.recording_host_path = os.path.join(abs_dpath_host, output_filename)

        try:
            # Get the directory part of the path and create it, including parents
            output_dir = os.path.dirname(container_recording_path)
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Ensured recording directory exists: {output_dir}")
        except OSError as e:
            return f"Error creating directory for recording: {e}"

        serial = self.get_device_serial()
        logger.info(f"Serial: {serial}")
        logger.info(f"Path: {container_recording_path}")
        logger.info(f"Bitrate: {bit_rate_mbps}")
        logger.info(f"Max Duration: {max_duration_seconds}")
        cmd = (
            f"scrcpy --serial {serial} "
            f"--record '{container_recording_path}' "
            f"--video-bit-rate {bit_rate_mbps}M "  # Set the bit rate
            f"--no-audio "  # Need this because the docker container version of the emulator doesn't have mic support
            f"--no-playback "
            f"--time-limit {max_duration_seconds} "  # Set the calculated time limit
            f"--show-touches"
        )
        logger.info(f"Attempting to run this command {cmd}")

        try:
            # Use shlex.split to parse the command and exec to run it directly
            self.recording_process = await asyncio.create_subprocess_exec(*shlex.split(cmd))
            logger.info(
                f"Started recording for device {serial}. Saving to {self.recording_host_path}"
            )
            return "Recording started."
        except Exception as e:
            logger.error(f"Failed to start recording process: {e}")
            return f"Error: Failed to start recording process: {e}"

    # CHANGED: Updated stop_recording method to use the host path in its message
    async def stop_recording(self) -> str:
        """Stops the current screen recording."""
        if not self.recording_process:
            return "Error: No recording is currently in progress."

        # Check if the process has already terminated
        if self.recording_process.returncode is None:
            # The process is still running, so we need to stop it.
            try:
                self.recording_process.send_signal(signal.SIGINT)
                await self.recording_process.wait()
                logger.info("Gracefully stopped recording process.")
            except ProcessLookupError:
                # Handle the rare case where the process terminates right after our check
                logger.warning("Recording process not found, it likely terminated on its own.")
            except Exception as e:
                logger.error(f"Failed to stop running recording process: {e}")
                return f"Error: Failed to stop recording: {e}"
        else:
            # The process had already finished.
            logger.info(
                f"Recording process already terminated with code: {self.recording_process.returncode}"
            )

        self.recording_process = None
        final_message = f"Recording stopped. Video saved to {self.recording_host_path}"
        logger.info(final_message)
        return final_message


if __name__ == "__main__":

    async def main():
        tools = AdbTools()

    asyncio.run(main())
