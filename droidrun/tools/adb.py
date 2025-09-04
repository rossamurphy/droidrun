"""
UI Actions - Core UI interaction tools for Android device control.
"""

import os
import io
import json
import shlex
import time
import logging
from llama_index.core.workflow import Context
from typing import Optional, Dict, Tuple, List, Any
from droidrun.agent.common.events import (
    InputTextActionEvent,
    KeyPressActionEvent,
    StartAppEvent,
    SwipeActionEvent,
    TapActionEvent,
    DragActionEvent,
)
from eval.android_env_client import AndroidEnvClient
from droidrun.tools.tools import Tools
from adbutils import adb
import requests
import base64
import asyncio
import math

logger = logging.getLogger("droidrun-tools")
PORTAL_DEFAULT_TCP_PORT = 8080

JEEVES_PACKAGE = "com.jeeves"
JEEVES_SERVICE = f"{JEEVES_PACKAGE}/.JeevesService"
FLASH_TOUCH_ACTION = "com.jeeves.FLASH_TOUCH"


class AdbTools(Tools):
    """Core UI interaction tools for Android device control."""

    def __init__(
        self,
        serial: str | None = None,
        use_tcp: bool = False,
        remote_tcp_port: int = PORTAL_DEFAULT_TCP_PORT,
        host_volume_path: str = "/opt/shared",
        container_mount_path: str = "opt/shared",
        client: Optional[AndroidEnvClient] = None,
        enable_jeeves: bool = True,
    ) -> None:
        """Initialize the AdbTools instance.

        Args:
            serial: Device serial number
            use_tcp: Whether to use TCP communication (default: False)
            tcp_port: TCP port for communication (default: 8080)
        """
        self.device = adb.device(serial=serial)
        self.use_tcp = use_tcp
        self.remote_tcp_port = remote_tcp_port
        self.tcp_forwarded = False

        self._ctx = None
        # Instance-level cache for clickable elements (index-based tapping)
        self.clickable_elements_cache: List[Dict[str, Any]] = []
        self.last_screenshot = None
        self.reason = None
        self.success = None
        self.finished = False
        # Memory storage for remembering important information
        self.memory: List[str] = []
        # Store all screenshots with timestamps
        self.screenshots: List[Dict[str, Any]] = []
        # Trajectory saving level
        self.save_trajectories = "none"

        self.recording_process: Optional[asyncio.subprocess.Process] = None
        self.recording_host_path: Optional[str] = None
        self.host_volume_path = os.path.abspath(host_volume_path)  # Resolve to an absolute path
        self.container_mount_path = container_mount_path

        self.enable_jeeves = enable_jeeves
        self.client = client
        self.serial = serial

        self.setup_keyboard()

        # Set up TCP forwarding if requested
        if self.use_tcp:
            self.setup_tcp_forward()

    def set_screenshot_save_dir(self, directory: str) -> None:
        """Set the directory where screenshots should be saved.

        Args:
            directory: Path to directory where screenshots will be saved
        """
        self.screenshot_save_dir = directory
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Screenshot save directory set to: {directory}")

    def setup_tcp_forward(self) -> bool:
        """
        Set up ADB TCP port forwarding for communication with the portal app.

        Returns:
            bool: True if forwarding was set up successfully, False otherwise
        """
        try:
            logger.debug(
                f"Setting up TCP port forwarding for port tcp:{self.remote_tcp_port} on device {self.device.serial}"
            )
            # Use adb forward command to set up port forwarding
            self.local_tcp_port = self.device.forward_port(self.remote_tcp_port)
            self.tcp_base_url = f"http://localhost:{self.local_tcp_port}"
            logger.debug(f"TCP port forwarding set up successfully to {self.tcp_base_url}")

            # Test the connection with a ping
            try:
                response = requests.get(f"{self.tcp_base_url}/ping", timeout=5)
                if response.status_code == 200:
                    logger.debug("TCP connection test successful")
                    self.tcp_forwarded = True
                    return True
                else:
                    logger.warning(
                        f"TCP connection test failed with status: {response.status_code}"
                    )
                    return False
            except requests.exceptions.RequestException as e:
                logger.warning(f"TCP connection test failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to set up TCP port forwarding: {e}")
            self.tcp_forwarded = False
            return False

    def teardown_tcp_forward(self) -> bool:
        """
        Remove ADB TCP port forwarding.

        Returns:
            bool: True if forwarding was removed successfully, False otherwise
        """
        try:
            if self.tcp_forwarded:
                logger.debug(f"Removing TCP port forwarding for port {self.local_tcp_port}")
                # remove forwarding
                cmd = f"killforward:tcp:{self.local_tcp_port}"
                logger.debug(f"Removing TCP port forwarding: {cmd}")
                c = self.device.open_transport(cmd)
                c.close()

                self.tcp_forwarded = False
                logger.debug("TCP port forwarding removed")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to remove TCP port forwarding: {e}")
            return False

    def setup_keyboard(self) -> bool:
        """
        Set up the DroidRun keyboard as the default input method.
        Simple setup that just switches to DroidRun keyboard without saving/restoring.

        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            self.device.shell("ime enable com.jeeves/.JeevesKeyboardIME")
            self.device.shell("ime set com.jeeves/.JeevesKeyboardIME")
            logger.debug("DroidRun keyboard setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup DroidRun keyboard: {e}")
            return False

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, "tcp_forwarded") and self.tcp_forwarded:
            self.teardown_tcp_forward()

    def _set_context(self, ctx: Context):
        self._ctx = ctx

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

    @Tools.ui_action
    def tap_by_index(self, index: int) -> str:
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
            # ALWAYS refresh elements first to ensure we tap on current screen state
            refresh_result = self.get_state()
            if refresh_result.get("error"):
                return f"Error: Could not refresh elements before tap: {refresh_result.get('error')}"
            
            if not self.clickable_elements_cache:
                return "Error: No UI elements found on current screen."

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

            logger.debug(f"Tapping element with index {index} at coordinates ({x}, {y})")
            # Get the device and tap at the coordinates
            self.device.click(x, y)
            logger.debug(f"Tapped element with index {index} at coordinates ({x}, {y})")

            # Emit coordinate action event for trajectory recording
            if self._ctx:
                element_text = element.get("text", "No text")
                element_class = element.get("className", "Unknown class")

                tap_event = TapActionEvent(
                    action_type="tap",
                    description=f"Tap element at index {index}: '{element_text}' ({element_class}) at coordinates ({x}, {y})",
                    x=x,
                    y=y,
                    element_index=index,
                    element_text=element_text,
                    element_bounds=bounds_str,
                )
                self._ctx.write_event_to_stream(tap_event)

            # Add a small delay to allow UI to update
            time.sleep(0.5)

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

            # Refresh element cache after tap to prepare for next operation
            try:
                refreshed_state = self.get_state()
                if refreshed_state.get("error"):
                    response_parts.append("Warning: Could not refresh element cache after tap")
                else:
                    response_parts.append(f"Cache updated: {len(self.clickable_elements_cache)} elements")
            except Exception as e:
                response_parts.append(f"Warning: Post-tap cache refresh failed: {str(e)}")

            return " | ".join(response_parts)
        except ValueError as e:
            return f"Error: {str(e)}"

    # Rename the old tap function to tap_by_coordinates for backward compatibility
    def tap_by_coordinates(self, x: int, y: int) -> bool:
        """
        Tap on the device screen at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Bool indicating success or failure
        """
        try:
            logger.debug(f"Tapping at coordinates ({x}, {y})")
            self.device.click(x, y)
            logger.debug(f"Tapped at coordinates ({x}, {y})")
            return True
        except ValueError as e:
            logger.debug(f"Error: {str(e)}")
            return False

    # Replace the old tap function with the new one
    def tap(self, index: int) -> str:
        """
        Tap on a UI element by its index.

        This function uses the cached clickable elements from the last get_clickables call
        to find the element with the given index and tap on its center coordinates.

        Args:
            index: Index of the element to tap

        Returns:
            Result message
        """
        return self.tap_by_index(index)

    @Tools.ui_action
    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> bool:
        """
        Performs a straight-line swipe gesture on the device screen.
        To perform a hold (long press), set the start and end coordinates to the same values and increase the duration as needed.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of swipe in seconds
        Returns:
            Bool indicating success or failure
        """
        try:
            if self._ctx:
                swipe_event = SwipeActionEvent(
                    action_type="swipe",
                    description=f"Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds",
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    duration_ms=duration_ms,
                )
                self._ctx.write_event_to_stream(swipe_event)

            self.device.swipe(start_x, start_y, end_x, end_y, float(duration_ms / 1000))
            time.sleep(duration_ms / 1000)
            logger.debug(
                f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds"
            )
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    @Tools.ui_action
    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 3
    ) -> bool:
        """
        Performs a straight-line drag and drop gesture on the device screen.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of drag in milliseconds
        Returns:
            Bool indicating success or failure
        """
        try:
            logger.debug(
                f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds"
            )
            self.device.drag(start_x, start_y, end_x, end_y, float(duration_ms / 1000))

            if self._ctx:
                drag_event = DragActionEvent(
                    action_type="drag",
                    description=f"Drag from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds",
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    duration_ms=duration_ms,
                )
                self._ctx.write_event_to_stream(drag_event)

            time.sleep(duration_ms / 1000)
            logger.debug(
                f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds"
            )
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    @Tools.ui_action
    def input_text(self, text: str) -> str:
        """
        Input text on the device.
        Always make sure that the Focused Element is not None before inputting text.

        Args:
            text: Text to input. Can contain spaces, newlines, and special characters including non-ASCII.

        Returns:
            Result message
        """
        try:
            if self.use_tcp and self.tcp_forwarded:
                # Use TCP communication
                encoded_text = base64.b64encode(text.encode()).decode()

                payload = {"base64_text": encoded_text}
                response = requests.post(
                    f"{self.tcp_base_url}/keyboard/input",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                logger.debug(
                    f"Keyboard input TCP response: {response.status_code}, {response.text}"
                )

                if response.status_code != 200:
                    return f"Error: HTTP request failed with status {response.status_code}: {response.text}"

            else:
                # Fallback to content provider method
                # Encode the text to Base64
                encoded_text = base64.b64encode(text.encode()).decode()

                # cmd = f'content insert --uri "content://com.jeeves/keyboard/input" --bind base64_text:s:"{encoded_text}"'
                # self.device.shell(cmd)

            if self._ctx:
                input_event = InputTextActionEvent(
                    action_type="input_text",
                    description=f"Input text: '{text[:50]}{'...' if len(text) > 50 else ''}'",
                    text=text,
                )
                self._ctx.write_event_to_stream(input_event)

            logger.debug(f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}")
            return f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"

        except requests.exceptions.RequestException as e:
            return f"Error: TCP request failed: {str(e)}"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error sending text input: {str(e)}"

    @Tools.ui_action
    def back(self) -> str:
        """
        Go back on the current view.
        This presses the Android back button.
        """
        try:
            logger.debug("Pressing key BACK")
            self.device.keyevent(4)

            if self._ctx:
                key_event = KeyPressActionEvent(
                    action_type="key_press",
                    description="Pressed key BACK",
                    keycode=4,
                    key_name="BACK",
                )
                self._ctx.write_event_to_stream(key_event)

            return "Pressed key BACK"
        except ValueError as e:
            return f"Error: {str(e)}"

    @Tools.ui_action
    def press_key(self, keycode: int) -> str:
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
            key_names = {
                66: "ENTER",
                4: "BACK",
                3: "HOME",
                67: "DELETE",
            }
            key_name = key_names.get(keycode, str(keycode))

            if self._ctx:
                key_event = KeyPressActionEvent(
                    action_type="key_press",
                    description=f"Pressed key {key_name}",
                    keycode=keycode,
                    key_name=key_name,
                )
                self._ctx.write_event_to_stream(key_event)

            logger.debug(f"Pressing key {key_name}")
            self.device.keyevent(keycode)
            logger.debug(f"Pressed key {key_name}")
            return f"Pressed key {key_name}"
        except ValueError as e:
            return f"Error: {str(e)}"

    @Tools.ui_action
    def start_app(self, package: str, activity: str | None = None) -> str:
        """
        Start an app on the device.

        Args:
            package: Package name (e.g., "com.android.settings")
            activity: Optional activity name
        """
        try:
            logger.debug(f"Starting app {package} with activity {activity}")
            if not activity:
                dumpsys_output = self.device.shell(
                    f"cmd package resolve-activity --brief {package}"
                )
                activity = dumpsys_output.splitlines()[1].split("/")[1]

            if self._ctx:
                start_app_event = StartAppEvent(
                    action_type="start_app",
                    description=f"Start app {package}",
                    package=package,
                    activity=activity,
                )
                self._ctx.write_event_to_stream(start_app_event)

            print(f"Activity: {activity}")

            self.device.app_start(package, activity)
            logger.debug(f"App started: {package} with activity {activity}")
            return f"App started: {package} with activity {activity}"
        except Exception as e:
            return f"Error: {str(e)}"

    def install_app(
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
            if not os.path.exists(apk_path):
                return f"Error: APK file not found at {apk_path}"

            logger.debug(
                f"Installing app: {apk_path} with reinstall: {reinstall} and grant_permissions: {grant_permissions}"
            )
            result = self.device.install(
                apk_path,
                nolaunch=True,
                uninstall=reinstall,
                flags=["-g"] if grant_permissions else [],
                silent=True,
            )
            logger.debug(f"Installed app: {apk_path} with result: {result}")
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    def take_screenshot(self, hide_overlay: bool = True) -> Tuple[str, bytes]:
        """
        Take a screenshot of the device.
        This function captures the current screen and adds the screenshot to context in the next message.
        Also stores the screenshot in the screenshots list with timestamp for later GIF creation.

        Args:
            hide_overlay: Whether to hide the overlay elements during screenshot (default: True)
        """
        try:
            logger.debug("Taking screenshot")
            img_format = "PNG"
            image_bytes = None

            if self.use_tcp and self.tcp_forwarded:
                # Add hideOverlay parameter to URL
                url = f"{self.tcp_base_url}/screenshot"
                if not hide_overlay:
                    url += "?hideOverlay=false"

                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    tcp_response = response.json()

                    # Check if response has the expected format with data field
                    if tcp_response.get("status") == "success" and "data" in tcp_response:
                        # Decode base64 string to bytes
                        base64_data = tcp_response["data"]
                        image_bytes = base64.b64decode(base64_data)
                        logger.debug("Screenshot taken via TCP")
                    else:
                        # Handle error response from server
                        error_msg = tcp_response.get("error", "Unknown error")
                        raise ValueError(f"Error taking screenshot via TCP: {error_msg}")
                else:
                    raise ValueError(f"Error taking screenshot via TCP: {response.status_code}")

            else:
                # Fallback to ADB screenshot method
                img = self.device.screenshot()
                img_buf = io.BytesIO()
                img.save(img_buf, format=img_format)
                image_bytes = img_buf.getvalue()
                logger.debug("Screenshot taken via ADB")

            # Store screenshot with timestamp
            self.screenshots.append(
                {
                    "timestamp": time.time(),
                    "image_data": image_bytes,
                    "format": img_format,
                }
            )
            return img_format, image_bytes

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error taking screenshot via TCP: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Error taking screenshot: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error taking screenshot: {str(e)}")

    def list_packages(self, include_system_apps: bool = False) -> List[str]:
        """
        List installed packages on the device.

        Args:
            include_system_apps: Whether to include system apps (default: False)

        Returns:
            List of package names
        """
        try:
            logger.debug("Listing packages")
            return self.device.list_packages(["-3"] if not include_system_apps else [])
        except ValueError as e:
            raise ValueError(f"Error listing packages: {str(e)}")

    @Tools.ui_action
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

    def remember(self, information: str) -> str:
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

    def _flash_touch_via_jeeves(self, x: int, y: int, execute_touch: bool = True):
        """Send flash-touch command via Jeeves."""
        if not self.enable_jeeves:
            return False

        try:
            cmd = f"am broadcast -a {FLASH_TOUCH_ACTION} --ei x {str(x)} --ei y {str(y)} --ez execute {str(execute_touch).lower()}"
            self.device.shell(cmd)

            logger.debug(f"🎩 Jeeves flash-touch: ({x}, {y}) execute={execute_touch}")
            return True

        except Exception as e:
            logger.debug(f"Error sending Jeeves flash-touch: {e}")
            return False

    def get_state(self, serial: Optional[str] = None) -> Dict[str, Any]:
        """
        Get both the a11y tree and phone state in a single call using the combined /state endpoint.

        Args:
            serial: Optional device serial number

        Returns:
            Dictionary containing both 'a11y_tree' and 'phone_state' data
        """

        try:
            logger.debug("Getting state")

            if self.use_tcp and self.tcp_forwarded:
                # Use TCP communication
                response = requests.get(f"{self.tcp_base_url}/state", timeout=10)

                if response.status_code == 200:
                    tcp_response = response.json()

                    # Check if response has the expected format
                    if isinstance(tcp_response, dict) and "data" in tcp_response:
                        data_str = tcp_response["data"]
                        try:
                            combined_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            return {
                                "error": "Parse Error",
                                "message": "Failed to parse JSON data from TCP response data field",
                            }
                    else:
                        # Fallback: assume direct JSON format
                        combined_data = tcp_response
                else:
                    return {
                        "error": "HTTP Error",
                        "message": f"HTTP request failed with status {response.status_code}",
                    }
            else:
                # Fallback to content provider method
                adb_output = self.device.shell(
                    "content query --uri content://com.jeeves/state",
                )

                state_data = self._parse_content_provider_output(adb_output)

                if state_data is None:
                    return {
                        "error": "Parse Error",
                        "message": "Failed to parse state data from ContentProvider response",
                    }

                if isinstance(state_data, dict):
                    data_str = None
                    if "data" in state_data:
                        data_str = state_data["data"]
                    elif "message" in state_data:
                        data_str = state_data["message"]

                    if data_str:
                        try:
                            combined_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            return {
                                "error": "Parse Error",
                                "message": "Failed to parse JSON data from ContentProvider response",
                            }
                    else:
                        return {
                            "error": "Format Error",
                            "message": "Neither 'data' nor 'message' field found in ContentProvider response",
                        }
                else:
                    return {
                        "error": "Format Error",
                        "message": f"Unexpected state data format: {type(state_data)}",
                    }

            # Validate that both a11y_tree and phone_state are present
            if "a11y_tree" not in combined_data:
                return {
                    "error": "Missing Data",
                    "message": "a11y_tree not found in combined state data",
                }

            if "phone_state" not in combined_data:
                return {
                    "error": "Missing Data",
                    "message": "phone_state not found in combined state data",
                }

            # Filter out the "type" attribute from all a11y_tree elements
            elements = combined_data["a11y_tree"]
            filtered_elements = []
            for element in elements:
                # Create a copy of the element without the "type" attribute
                filtered_element = {k: v for k, v in element.items() if k != "type"}

                # Also filter children if present
                if "children" in filtered_element:
                    filtered_element["children"] = [
                        {k: v for k, v in child.items() if k != "type"}
                        for child in filtered_element["children"]
                    ]

                filtered_elements.append(filtered_element)

            self.clickable_elements_cache = filtered_elements

            return {
                "a11y_tree": filtered_elements,
                "phone_state": combined_data["phone_state"],
            }

        except requests.exceptions.RequestException as e:
            return {
                "error": "TCP Error",
                "message": f"TCP request failed: {str(e)}",
            }
        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error getting combined state: {str(e)}",
            }

    def ping(self) -> Dict[str, Any]:
        """
        Test the TCP connection using the /ping endpoint.

        Returns:
            Dictionary with ping result
        """
        try:
            if self.use_tcp and self.tcp_forwarded:
                response = requests.get(f"{self.tcp_base_url}/ping", timeout=5)

                if response.status_code == 200:
                    try:
                        tcp_response = response.json() if response.content else {}
                        logger.debug(f"Ping TCP response: {tcp_response}")
                        return {
                            "status": "success",
                            "message": "Ping successful",
                            "response": tcp_response,
                        }
                    except json.JSONDecodeError:
                        return {
                            "status": "success",
                            "message": "Ping successful (non-JSON response)",
                            "response": response.text,
                        }
                else:
                    return {
                        "status": "error",
                        "message": f"Ping failed with status {response.status_code}: {response.text}",
                    }
            else:
                return {
                    "status": "error",
                    "message": "TCP communication is not enabled",
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Ping failed: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during ping: {str(e)}",
            }

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

        logger.info(f"Serial: {self.serial}")
        logger.info(f"Path: {container_recording_path}")
        logger.info(f"Bitrate: {bit_rate_mbps}")
        logger.info(f"Max Duration: {max_duration_seconds}")
        cmd = (
            f"scrcpy --serial {self.serial} "
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
                f"Started recording for device {self.serial}. Saving to {self.recording_host_path}"
            )
            return "Recording started."
        except Exception as e:
            logger.error(f"Failed to start recording process: {e}")
            return f"Error: Failed to start recording process: {e}"

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


def _shell_test_cli(serial: str, command: str) -> tuple[str, float]:
    """
    Run an adb shell command using the adb CLI and measure execution time.
    Args:
        serial: Device serial number
        command: Shell command to run
    Returns:
        Tuple of (output, elapsed_time)
    """
    import time
    import subprocess

    adb_cmd = ["adb", "-s", serial, "shell", command]
    start = time.perf_counter()
    result = subprocess.run(adb_cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    output = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    return output, elapsed


def _shell_test():
    device = adb.device("emulator-5554")
    # Native Python adb client
    start = time.time()
    res = device.shell("echo 'Hello, World!'")
    end = time.time()
    print(f"[Native] Shell execution took {end - start:.3f} seconds: {res}")

    start = time.time()
    res = device.shell("content query --uri content://com.jeeves/state")
    end = time.time()
    print(f"[Native] Shell execution took {end - start:.3f} seconds: phone_state")

    # CLI version
    output, elapsed = _shell_test_cli("emulator-5554", "echo 'Hello, World!'")
    print(f"[CLI] Shell execution took {elapsed:.3f} seconds: {output}")

    output, elapsed = _shell_test_cli(
        "emulator-5554", "content query --uri content://com.jeeves/state"
    )
    print(f"[CLI] Shell execution took {elapsed:.3f} seconds: phone_state")


def _list_packages():
    tools = AdbTools()
    print(tools.list_packages())


def _start_app():
    tools = AdbTools()
    tools.start_app("com.android.settings", ".Settings")


if __name__ == "__main__":
    _start_app()
