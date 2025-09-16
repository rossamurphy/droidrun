import base64
import inspect
import logging
import re
from typing import List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock

from droidrun.agent.context import Reflection

logger = logging.getLogger("droidrun")


def message_copy(message: ChatMessage, deep=True) -> ChatMessage:
    if deep:
        copied_message = message.model_copy()
        copied_message.blocks = [block.model_copy() for block in message.blocks]

        return copied_message
    copied_message = message.model_copy()

    # Create a new, independent list containing the same block references
    copied_message.blocks = list(message.blocks)  # or original_message.blocks[:]

    return copied_message


async def add_reflection_summary(
    reflection: Reflection, chat_history: List[ChatMessage]
) -> List[ChatMessage]:
    """Add reflection summary and advice to help the planner understand what went wrong and what to do differently."""

    reflection_text = "\n### The last task failed. You have additional information about what happened. \nThe Reflection from Previous Attempt:\n"

    if reflection.summary:
        reflection_text += f"**What happened:** {reflection.summary}\n\n"

    if reflection.advice:
        reflection_text += f"**Recommended approach for this retry:** {reflection.advice}\n"

    reflection_block = TextBlock(text=reflection_text)

    # Copy chat_history and append reflection block to the last message
    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(reflection_block)

    return chat_history


def _format_ui_elements(ui_data, level=0) -> str:
    """Format UI elements in natural language: index. className: resourceId, text - bounds"""
    if not ui_data:
        return ""

    formatted_lines = []
    indent = "  " * level  # Indentation for nested elements

    # Handle both list and single element
    elements = ui_data if isinstance(ui_data, list) else [ui_data]

    for element in elements:
        if not isinstance(element, dict):
            continue

        # Extract element properties
        index = element.get("index", "")
        class_name = element.get("className", "")
        resource_id = element.get("resourceId", "")
        text = element.get("text", "")
        bounds = element.get("bounds", "")
        children = element.get("children", [])

        # Format the line: index. className: resourceId, text - bounds
        line_parts = []
        if index != "":
            line_parts.append(f"{index}.")
        if class_name:
            line_parts.append(class_name + ":")

        details = []
        if resource_id:
            details.append(f'"{resource_id}"')
        if text:
            details.append(f'"{text}"')
        if details:
            line_parts.append(", ".join(details))

        if bounds:
            line_parts.append(f"- ({bounds})")

        formatted_line = f"{indent}{' '.join(line_parts)}"
        formatted_lines.append(formatted_line)

        # Recursively format children with increased indentation
        if children:
            child_formatted = _format_ui_elements(children, level + 1)
            if child_formatted:
                formatted_lines.append(child_formatted)

    return "\n".join(formatted_lines)


from .ui_formatting import format_ui_elements_as_text


async def add_ui_text_block(
    ui_state: str, chat_history: List[ChatMessage], copy=True
) -> List[ChatMessage]:
    """Add UI elements to the chat history using unified formatting."""
    if ui_state:
        # Use the shared formatting function for consistency with training data
        formatted_ui = format_ui_elements_as_text(ui_state)
        ui_block = TextBlock(text=f"\n{formatted_ui}\n")

        if copy:
            chat_history = chat_history.copy()
            chat_history[-1] = message_copy(chat_history[-1])
        chat_history[-1].blocks.append(ui_block)
    return chat_history


async def add_screenshot_image_block(
    screenshot, chat_history: List[ChatMessage], copy=True
) -> None:
    if screenshot:
        image_block = ImageBlock(image=base64.b64encode(screenshot))
        if copy:
            chat_history = (
                chat_history.copy()
            )  # Create a copy of chat history to avoid modifying the original
            chat_history[-1] = message_copy(chat_history[-1])
        chat_history[-1].blocks.append(image_block)
    return chat_history


async def add_phone_state_block(phone_state, chat_history: List[ChatMessage]) -> List[ChatMessage]:
    # Format the phone state data nicely
    if isinstance(phone_state, dict) and "error" not in phone_state:
        current_app = phone_state.get("currentApp", "")
        package_name = phone_state.get("packageName", "Unknown")
        keyboard_visible = phone_state.get("keyboardVisible", False)
        focused_element = phone_state.get("focusedElement")

        # Format the focused element
        if focused_element:
            element_text = focused_element.get("text", "")
            element_class = focused_element.get("className", "")
            element_resource_id = focused_element.get("resourceId", "")

            # Build focused element description
            focused_desc = f"'{element_text}' {element_class}"
            if element_resource_id:
                focused_desc += f" | ID: {element_resource_id}"
        else:
            focused_desc = "None"

        phone_state_text = f"""
**Current Phone State:**
• **App:** {current_app} ({package_name})
• **Keyboard:** {"Visible" if keyboard_visible else "Hidden"}
• **Focused Element:** {focused_desc}
        """
    else:
        # Handle error cases or malformed data
        if isinstance(phone_state, dict) and "error" in phone_state:
            phone_state_text = (
                f"\n📱 **Phone State Error:** {phone_state.get('message', 'Unknown error')}\n"
            )
        else:
            phone_state_text = f"\n📱 **Phone State:** {phone_state}\n"

    ui_block = TextBlock(text=phone_state_text)
    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(ui_block)
    return chat_history


async def add_packages_block(packages, chat_history: List[ChatMessage]) -> List[ChatMessage]:
    ui_block = TextBlock(text=f"\nInstalled packages: {packages}\n```\n")
    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(ui_block)
    return chat_history


async def add_memory_block(memory: List[str], chat_history: List[ChatMessage]) -> List[ChatMessage]:
    memory_block = "\n### Remembered Information:\n"
    for idx, item in enumerate(memory, 1):
        memory_block += f"{idx}. {item}\n"

    for i, msg in enumerate(chat_history):
        if msg.role == "user":
            if isinstance(msg.content, str):
                updated_content = f"{memory_block}\n\n{msg.content}"
                chat_history[i] = ChatMessage(role="user", content=updated_content)
            elif isinstance(msg.content, list):
                memory_text_block = TextBlock(text=memory_block)
                content_blocks = [memory_text_block] + msg.content
                chat_history[i] = ChatMessage(role="user", content=content_blocks)
            break
    return chat_history


def estimate_chat_tokens(chat_history: List[ChatMessage]) -> int:
    """
    Estimate total tokens in chat history using word-based approximation.
    Rough approximation: 1 token ≈ 0.75 words, so 4/3 words per token
    """
    total_words = 0

    for msg in chat_history:
        if isinstance(msg.content, str):
            total_words += len(msg.content.split())
        elif isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    total_words += len(block.text.split())

    return int(total_words * 4 / 3)


def should_force_summarization(
    chat_history: List[ChatMessage], token_threshold: int = 24000
) -> bool:
    """
    Check if chat history exceeds token threshold and needs summarization.
    Default threshold: 24k tokens (18k words)
    """
    return estimate_chat_tokens(chat_history) > token_threshold


def create_summarization_prompt(chat_history: List[ChatMessage], task_goal: str) -> str:
    """
    Create a prompt that forces the agent to summarize conversation history
    while preserving task-relevant information.
    """
    return f"""
CONTEXT LENGTH MANAGEMENT REQUIRED

Your conversation history has grown too long ({estimate_chat_tokens(chat_history):,} tokens). You must now summarize the important information from your previous actions and observations to continue efficiently.

TASK REMINDER: {task_goal}

Please use the remember() tool to save a concise summary that includes:
1. What you've accomplished so far toward the task goal
2. Current state/location in the app/interface  
3. Any important observations about the UI or app behavior
4. What still needs to be done to complete the task
5. Any errors or challenges encountered

Your summary should be 2-3 sentences maximum but capture everything essential for completing the task.

IMPORTANT: You MUST call remember() with your summary before taking any other action.
"""


async def handle_context_length_management(
    chat_history: List[ChatMessage], task_goal: str, keep_recent_turns: int = 3
) -> tuple[List[ChatMessage], bool]:
    """
    Handle context length management by forcing summarization when needed.

    Returns:
        - Modified chat history
        - Boolean indicating if summarization was triggered
    """
    if not should_force_summarization(chat_history):
        return chat_history, False

    # Create summarization prompt
    summarization_prompt = create_summarization_prompt(chat_history, task_goal)

    # Add forced summarization message
    summarization_msg = ChatMessage(role="system", content=summarization_prompt)

    # Keep system message, recent turns, and add summarization prompt
    system_msgs = [msg for msg in chat_history if msg.role == "system"]
    recent_msgs = (
        chat_history[-keep_recent_turns:] if len(chat_history) > keep_recent_turns else chat_history
    )

    new_history = system_msgs + recent_msgs + [summarization_msg]

    return new_history, True


async def get_reflection_block(reflections: List[Reflection]) -> ChatMessage:
    reflection_block = "\n### You also have additional Knowledge to help you guide your current task from previous experiences:\n"
    for reflection in reflections:
        reflection_block += f"**{reflection.advice}\n"

    return ChatMessage(role="user", content=reflection_block)


async def add_task_history_block(
    completed_tasks: list[dict], failed_tasks: list[dict], chat_history: List[ChatMessage]
) -> List[ChatMessage]:
    task_history = ""

    # Combine all tasks and show in chronological order
    all_tasks = completed_tasks + failed_tasks

    if all_tasks:
        task_history += "Task History (chronological order):\n"
        for i, task in enumerate(all_tasks, 1):
            if hasattr(task, "description"):
                status_indicator = (
                    "[success]"
                    if hasattr(task, "status") and task.status == "completed"
                    else "[failed]"
                )
                task_history += f"{i}. {status_indicator} {task.description}\n"
            elif isinstance(task, dict):
                # For backward compatibility with dict format
                task_description = task.get("description", str(task))
                status_indicator = "[success]" if task in completed_tasks else "[failed]"
                task_history += f"{i}. {status_indicator} {task_description}\n"
            else:
                status_indicator = "[success]" if task in completed_tasks else "[failed]"
                task_history += f"{i}. {status_indicator} {task}\n"

    task_block = TextBlock(text=f"{task_history}")

    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(task_block)
    return chat_history


def parse_tool_descriptions(tool_list) -> str:
    """Parses the available tools and their descriptions for the system prompt."""
    logger.info("🛠️  Parsing tool descriptions...")
    tool_descriptions = []

    for tool in tool_list.values():
        assert callable(tool), f"Tool {tool} is not callable."
        tool_name = tool.__name__
        tool_signature = inspect.signature(tool)
        tool_docstring = tool.__doc__ or "No description available."
        formatted_signature = f'def {tool_name}{tool_signature}:\n    """{tool_docstring}"""\n...'
        tool_descriptions.append(formatted_signature)
        logger.debug(f"  - Parsed tool: {tool_name}")
    descriptions = "\n".join(tool_descriptions)
    logger.info(f"🔩 Found {len(tool_descriptions)} tools.")
    return descriptions


def parse_persona_description(personas) -> str:
    """Parses the available agent personas and their descriptions for the system prompt."""
    logger.debug("👥 Parsing agent persona descriptions for Planner Agent...")

    if not personas:
        logger.warning("No agent personas provided to Planner Agent")
        return "No specialized agents available."

    persona_descriptions = []
    for persona in personas:
        # Format each persona with name, description, and expertise areas
        expertise_list = (
            ", ".join(persona.expertise_areas) if persona.expertise_areas else "General tasks"
        )
        formatted_persona = (
            f"- **{persona.name}**: {persona.description}\n  Expertise: {expertise_list}"
        )
        persona_descriptions.append(formatted_persona)
        logger.debug(f"  - Parsed persona: {persona.name}")

    # Join all persona descriptions into a single string
    descriptions = "\n".join(persona_descriptions)
    logger.debug(f"👤 Found {len(persona_descriptions)} agent personas.")
    return descriptions


def extract_code_and_thought(response_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts code from Markdown blocks (```python ... ```) and the surrounding text (thought),
    handling indented code blocks.

    Returns:
        Tuple[Optional[code_string], thought_string]
    """
    logger.debug("✂️ Extracting code and thought from response...")
    code_pattern = r"^\s*```python\s*\n(.*?)\n^\s*```\s*?$"
    code_matches = list(re.finditer(code_pattern, response_text, re.DOTALL | re.MULTILINE))

    if not code_matches:
        logger.debug("  - No code block found. Entire response is thought.")
        return None, response_text.strip()

    extracted_code_parts = []
    for match in code_matches:
        code_content = match.group(1)
        extracted_code_parts.append(code_content)

    extracted_code = "\n\n".join(extracted_code_parts)

    thought_parts = []
    last_end = 0
    for match in code_matches:
        start, end = match.span(0)
        thought_parts.append(response_text[last_end:start])
        last_end = end
    thought_parts.append(response_text[last_end:])

    thought_text = "".join(thought_parts).strip()
    thought_preview = (thought_text[:100] + "...") if len(thought_text) > 100 else thought_text
    logger.debug(f"  - Extracted thought: {thought_preview}")

    return extracted_code, thought_text


def normalize_conversation(
    messages: List[ChatMessage], max_tokens: Optional[int] = None
) -> List[ChatMessage]:
    """
    Normalize conversation to ensure proper structure:
    1. System message first (required)
    2. Strict user/assistant alternation
    3. Respect token limits while preserving system message and recent context

    Args:
        messages: Raw conversation messages
        max_tokens: Maximum tokens to allow (approximate word count * 1.3)

    Returns:
        Normalized conversation with system message first, then alternating user/assistant
    """
    if not messages:
        raise ValueError("Messages cannot be empty - system message is required")

    # Separate system messages from conversation messages
    system_messages = [msg for msg in messages if msg.role == "system"]
    conversation_messages = [msg for msg in messages if msg.role != "system"]

    if not system_messages:
        raise ValueError("System message is required but not found")

    # Use the first system message (merge multiple if they exist)
    if len(system_messages) > 1:
        merged_system_content = "\n\n".join(msg.content for msg in system_messages)
        system_message = ChatMessage(role="system", content=merged_system_content)
    else:
        system_message = system_messages[0]

    # Normalize conversation alternation
    normalized_conversation = ensure_alternation(conversation_messages)

    # Handle token limits if specified
    if max_tokens:
        normalized_conversation = truncate_conversation(
            normalized_conversation, max_tokens, system_message
        )

    # Return: system message first, then normalized conversation
    return [system_message] + normalized_conversation


def ensure_alternation(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Ensure strict user/assistant alternation by merging consecutive same-role messages."""
    if not messages:
        return messages

    normalized = []

    for msg in messages:
        if not normalized:
            normalized.append(msg)
        elif normalized[-1].role == msg.role:
            # Merge consecutive messages from same role
            merged_content = f"{normalized[-1].content}\n\n{msg.content}"
            normalized[-1] = ChatMessage(role=msg.role, content=merged_content)
        else:
            normalized.append(msg)

    # Ensure conversation starts with user message
    if normalized and normalized[0].role != "user":
        # Insert generic user message at the beginning
        generic_user = ChatMessage(role="user", content="Please help me with this task.")
        normalized.insert(0, generic_user)

    # Final pass: ensure perfect alternation
    final_normalized = []
    expected_role = "user"

    for msg in normalized:
        if msg.role == expected_role:
            final_normalized.append(msg)
            expected_role = "assistant" if expected_role == "user" else "user"
        else:
            # Add minimal bridge message to maintain alternation
            if expected_role == "user":
                bridge_msg = ChatMessage(role="user", content="Continue.")
            else:
                bridge_msg = ChatMessage(role="assistant", content="Understood.")

            final_normalized.append(bridge_msg)
            final_normalized.append(msg)
            expected_role = "assistant" if msg.role == "user" else "user"

    return final_normalized


def truncate_conversation(
    conversation: List[ChatMessage], max_tokens: int, system_message: ChatMessage
) -> List[ChatMessage]:
    """
    Truncate conversation to fit token limit while preserving recent context.
    Strategy: Keep recent messages and ensure user/assistant alternation is maintained.
    """

    # Rough token estimation (words * 1.3)
    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    system_tokens = estimate_tokens(system_message.content)
    available_tokens = max_tokens - system_tokens

    if available_tokens <= 0:
        logger.warning("System message exceeds token limit - proceeding anyway")
        return conversation[:2] if len(conversation) >= 2 else conversation

    # Work backwards from the most recent messages
    truncated = []
    current_tokens = 0

    # Always try to keep at least the last user-assistant pair
    for msg in reversed(conversation):
        msg_tokens = estimate_tokens(msg.content)

        if current_tokens + msg_tokens <= available_tokens:
            truncated.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break

    # Ensure we have at least some conversation and proper alternation
    if not truncated and conversation:
        # Take the most recent message even if it exceeds limit
        truncated = [conversation[-1]]

    # Ensure truncated conversation starts with user message for proper alternation
    if truncated and truncated[0].role != "user":
        if len(truncated) > 1:
            # Remove first assistant message to maintain alternation
            truncated = truncated[1:]
        else:
            # Replace with a generic user message
            truncated = [ChatMessage(role="user", content="Please help with this task.")]

    if len(truncated) != len(conversation):
        logger.info(
            f"Truncated conversation from {len(conversation)} to {len(truncated)} messages to fit token limit"
        )

    return truncated
