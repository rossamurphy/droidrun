import logging
import re
import time
import asyncio
import json
import os
from typing import List, Optional, Tuple, Union
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.memory import Memory
from droidrun.agent.codeact.events import (
    TaskInputEvent,
    TaskEndEvent,
    TaskExecutionEvent,
    TaskExecutionResultEvent,
    TaskThinkingEvent,
    EpisodicMemoryEvent,
)
from droidrun.agent.common.events import ScreenshotEvent
from droidrun.agent.utils import chat_utils
from droidrun.agent.utils.executer import SimpleCodeExecutor
from droidrun.agent.codeact.prompts import (
    DEFAULT_CODE_ACT_USER_PROMPT,
    DEFAULT_NO_THOUGHTS_PROMPT,
)

from droidrun.agent.context.episodic_memory import EpisodicMemory, EpisodicMemoryStep
from droidrun.tools import Tools
from typing import Optional, Dict, Tuple, List, Any, Callable
from droidrun.agent.context.agent_persona import AgentPersona

logger = logging.getLogger("droidrun")


class CodeActAgent(Workflow):
    """
    An agent that uses a ReAct-like cycle (Thought -> Code -> Observation)
    to solve problems requiring code execution. It extracts code from
    Markdown blocks and uses specific step types for tracking.
    """

    def __init__(
        self,
        llm: LLM,
        persona: AgentPersona,
        vision: bool,
        tools_instance: "Tools",
        all_tools_list: Dict[str, Callable[..., Any]],
        max_steps: int = 5,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        # assert instead of if
        assert llm, "llm must be provided."
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.max_steps = max_steps

        self.user_prompt = persona.user_prompt
        self.no_thoughts_prompt = None

        self.vision = vision

        self.chat_memory = None
        self.episodic_memory = EpisodicMemory(persona=persona)
        self.remembered_info = None

        self.goal = None
        self.steps_counter = 0
        self.code_exec_counter = 0
        self.debug = debug

        self.tools = tools_instance

        self.tool_list = {}

        for tool_name in persona.allowed_tools:
            if tool_name in all_tools_list:
                self.tool_list[tool_name] = all_tools_list[tool_name]

        self.tool_descriptions = chat_utils.parse_tool_descriptions(self.tool_list)
        self.persona = persona  # Store persona for later system prompt formatting
        
        # System prompt will be created in prepare_chat once we have the goal
        self.system_prompt = None

        self.required_context = persona.required_context
        
        # Track incremental progress summary instead of full history
        self.progress_summary = ""
        self.current_task = None

        self.executor = SimpleCodeExecutor(
            loop=asyncio.get_event_loop(),
            locals={},
            tools=self.tool_list,
            globals={"__builtins__": __builtins__},
        )

        logger.info("✅ CodeActAgent initialized successfully.")

    def _update_progress_summary(self, action_desc: str, result: str) -> None:
        """
        Incrementally update progress summary based on new action.
        Fast O(1) operation that maintains a rolling summary of what's happening.
        
        Args:
            action_desc: Description of the action taken
            result: Result/output from the action
        """
        # Smart incremental summary update
        if not self.progress_summary:
            # First action - establish baseline
            self.progress_summary = f"Started task. Attempted: {action_desc}"
        else:
            # Update summary based on what we learned
            if "tap" in action_desc.lower():
                if "error" in result.lower() or "failed" in result.lower():
                    self.progress_summary += f" → Tap failed, trying different approach"
                else:
                    self.progress_summary += f" → Tapped UI element"
            elif "scroll" in action_desc.lower() or "swipe" in action_desc.lower():
                self.progress_summary += f" → Navigated screen"
            elif "type" in action_desc.lower() or "input" in action_desc.lower():
                self.progress_summary += f" → Entered text"
            elif "launch" in action_desc.lower():
                self.progress_summary += f" → Opened app"
            elif "get_state" in action_desc.lower():
                self.progress_summary += f" → Analyzed UI"
            else:
                # Generic update
                self.progress_summary += f" → {action_desc}"
        
        # Keep summary concise - truncate if too long
        if len(self.progress_summary) > 200:
            # Keep the task start and recent progress
            parts = self.progress_summary.split(" → ")
            if len(parts) > 3:
                self.progress_summary = parts[0] + " → ... → " + " → ".join(parts[-2:])
        
    def _get_progress_context(self) -> str:
        """Get current progress summary for context."""
        if not self.progress_summary:
            return ""
        return f"Progress so far: {self.progress_summary}"
    
    def _extract_action_description(self, code: str) -> str:
        """
        Extract a human-readable description from code.
        
        Args:
            code: The Python code to analyze
            
        Returns:
            A brief description of what the code does
        """
        # Look for common patterns in the code
        if "tap(" in code or "tap_by_index(" in code:
            # Extract what was tapped
            match = re.search(r'tap(?:_by_index)?\((.*?)\)', code)
            if match:
                return f"Tap {match.group(1)}"
        elif "swipe(" in code:
            return "Swipe gesture"
        elif "scroll(" in code:
            return "Scroll"
        elif "type_text(" in code or "input_text(" in code:
            match = re.search(r'(?:type_text|input_text)\((.*?)\)', code)
            if match:
                return f"Type text: {match.group(1)[:30]}"
        elif "launch_app(" in code:
            match = re.search(r'launch_app\((.*?)\)', code)
            if match:
                return f"Launch app: {match.group(1)}"
        elif "press_key(" in code:
            match = re.search(r'press_key\((.*?)\)', code)
            if match:
                return f"Press key: {match.group(1)}"
        elif "complete(" in code:
            return "Complete task"
        elif "get_state(" in code:
            return "Get UI state"
        elif "take_screenshot(" in code:
            return "Take screenshot"
        else:
            # Fallback: use first line of code
            first_line = code.split('\n')[0][:50]
            return first_line if first_line else "Execute code"

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> TaskInputEvent:
        """Prepare chat history from user input."""
        logger.info("💬 Preparing chat for task execution...")

        self.chat_memory: Memory = await ctx.get("chat_memory", default=Memory.from_defaults())

        user_input = ev.get("input", default=None)
        assert user_input, "User input cannot be empty."

        if ev.remembered_info:
            self.remembered_info = ev.remembered_info

        logger.debug("  - Adding goal to memory.")
        goal = user_input
        self.current_task = goal  # Store for context in future messages
        
        # Create system prompt with task included to ensure it's never truncated
        if not self.system_prompt:
            self.system_prompt_content = self.persona.system_prompt.format(
                tool_descriptions=self.tool_descriptions,
                current_task=goal  # Add task to system prompt
            )
            self.system_prompt = ChatMessage(role="system", content=self.system_prompt_content)
        
        self.user_message = ChatMessage(
            role="user",
            content=PromptTemplate(self.user_prompt or DEFAULT_CODE_ACT_USER_PROMPT).format(
                goal=goal
            ),
        )
        self.no_thoughts_prompt = ChatMessage(
            role="assistant",
            content=PromptTemplate(DEFAULT_NO_THOUGHTS_PROMPT).format(goal=goal),
        )

        await self.chat_memory.aput(self.user_message)

        await ctx.set("chat_memory", self.chat_memory)
        input_messages = self.chat_memory.get_all()
        return TaskInputEvent(input=input_messages)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: TaskInputEvent
    ) -> TaskThinkingEvent | TaskEndEvent:
        """Handle LLM input."""
        chat_history = ev.input
        assert len(chat_history) > 0, "Chat history cannot be empty."
        ctx.write_event_to_stream(ev)

        if self.steps_counter >= self.max_steps:
            ev = TaskEndEvent(
                success=False,
                reason=f"Reached max step count of {self.max_steps} steps",
            )
            ctx.write_event_to_stream(ev)
            return ev

        self.steps_counter += 1
        logger.info(f"🧠 Step {self.steps_counter}: Thinking...")

        model = self.llm.class_name()

        if "remember" in self.tool_list and self.remembered_info:
            await ctx.set("remembered_info", self.remembered_info)
            chat_history = await chat_utils.add_memory_block(self.remembered_info, chat_history)

        logger.info(f"🔍 Required contexts: {self.required_context}")
        for context in self.required_context:
            logger.info(f"🔍 Processing context: {context}")
            if model == "DeepSeek":
                logger.warning("[yellow]DeepSeek doesnt support images. Disabling screenshots[/]")

            elif self.vision == True and context == "screenshot":
                screenshot = (await self.tools.take_annotated_screenshot())[1]
                ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))

                await ctx.set("screenshot", screenshot)
                chat_history = await chat_utils.add_screenshot_image_block(screenshot, chat_history)

            if context == "ui_state":
                logger.info(f"🔍 Processing ui_state context...")
                try:
                    state = await self.tools.get_state()
                    a11y_tree = state.get('a11y_tree', '{}')
                    logger.info(f"🔍 Got UI state type: {type(a11y_tree)}")
                    if isinstance(a11y_tree, str):
                        logger.info(f"🔍 UI state string length: {len(a11y_tree)}")
                        logger.info(f"🔍 UI state preview: {a11y_tree[:100]}")
                    else:
                        logger.info(f"🔍 UI state is a dict/list with {len(str(a11y_tree))} chars when stringified")
                    await ctx.set("ui_state", a11y_tree)
                    chat_history = await chat_utils.add_ui_text_block(
                        a11y_tree, chat_history
                    )
                    logger.info(f"🔍 Added UI text block to chat history")
                    chat_history = await chat_utils.add_phone_state_block(
                        state["phone_state"], chat_history
                    )
                    logger.info(f"🔍 Added phone state block to chat history")
                except Exception as e:
                    logger.warning(f"Exception Raised: {e}")
                    logger.warning(
                        f"⚠️ Error retrieving state from the connected device. Is the Accessibility Service enabled?"
                    )

            if context == "ui_cache_only":
                # Cache UI elements for tap_by_index without sending text to LLM
                try:
                    state = await self.tools.get_state()
                    await ctx.set("ui_state", state["a11y_tree"])
                    # Don't add to chat_history - just cache the elements
                except Exception as e:
                    logger.warning(f"Exception Raised: {e}")
                    logger.warning(
                        f"⚠️ Error retrieving state from the connected device. Is the Accessibility Service enabled?"
                    )

            if context == "packages":
                chat_history = await chat_utils.add_packages_block(
                    await self.tools.list_packages(include_system_apps=True),
                    chat_history,
                )

        response = await self._get_llm_response(ctx, chat_history, debug=self.debug)
        if response is None:
            return TaskEndEvent(
                success=False, reason="LLM response is None. This is a critical error."
            )

        await self.chat_memory.aput(response.message)

        code, thoughts = chat_utils.extract_code_and_thought(response.message.content)

        event = TaskThinkingEvent(thoughts=thoughts, code=code)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_llm_output(
        self, ctx: Context, ev: TaskThinkingEvent
    ) -> Union[TaskExecutionEvent, TaskInputEvent]:
        """Handle LLM output."""
        logger.debug("⚙️ Handling LLM output...")
        code = ev.code
        thoughts = ev.thoughts

        # removing this hard constraint to output thoughts
        # if you get code, just execute it!

        # if not thoughts:
        #     logger.warning(
        #         "🤔 LLM provided code without thoughts. Adding reminder prompt."
        #     )
        #     await self.chat_memory.aput(self.no_thoughts_prompt)
        # else:
        #     logger.info(f"🤔 Reasoning: {thoughts}")

        if code:
            return TaskExecutionEvent(code=code)
        else:
            message = ChatMessage(
                role="user",
                content="No code was provided. If you want to mark task as complete (whether it failed or succeeded), use complete(success:bool, reason:str) function within a code block ```python\n```.",
            )
            await self.chat_memory.aput(message)
            return TaskInputEvent(input=self.chat_memory.get_all())

    @step
    async def execute_code(
        self, ctx: Context, ev: TaskExecutionEvent
    ) -> Union[TaskExecutionResultEvent, TaskEndEvent]:
        """Execute the code and return the result."""
        code = ev.code
        assert code, "Code cannot be empty."
        # CLEAN ACTION EXECUTION LOGGING
        print("\n" + "⚡"*60)
        print("🛠️  EXECUTING ACTION")
        print("⚡"*60)
        print(f"📝 Code:\n```python\n{code}\n```")
        print("-" * 60)
        print("⏳ Running action...")
        
        logger.info(f"⚡ Executing action...")
        logger.debug(f"Code to execute:\n```python\n{code}\n```")

        try:
            self.code_exec_counter += 1
            result = await self.executor.execute(ctx, code)
            
            # CLEAN ACTION RESULT LOGGING
            result_text = str(result) if result else "No output"
            displayed_result = result_text[:200] + "..." if len(result_text) > 200 else result_text
            print(f"✅ ACTION RESULT: {displayed_result}")
            print("⚡"*60 + "\n")
            
            logger.info(f"💡 Code execution successful. Result: {result}")
            
            # Update incremental progress summary
            action_desc = self._extract_action_description(code)
            self._update_progress_summary(action_desc, str(result) if result else "")

            if self.tools.finished == True:
                logger.debug("  - Task completed.")
                event = TaskEndEvent(success=self.tools.success, reason=self.tools.reason)
                ctx.write_event_to_stream(event)
                return event

            self.remembered_info = self.tools.memory

            event = TaskExecutionResultEvent(output=str(result))
            ctx.write_event_to_stream(event)
            return event

        except Exception as e:
            # CLEAN ERROR LOGGING
            error_text = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            print(f"❌ ACTION FAILED: {error_text}")
            print("⚡"*60 + "\n")
            
            logger.error(f"💥 Action failed: {e}")
            if self.debug:
                logger.error("Exception details:", exc_info=True)
            error_message = f"Error during execution: {e}"
            
            # Update incremental progress summary with error info
            action_desc = self._extract_action_description(code)
            self._update_progress_summary(action_desc, f"Error: {str(e)}")

            event = TaskExecutionResultEvent(output=error_message)
            ctx.write_event_to_stream(event)
            return event

    @step
    async def handle_execution_result(
        self, ctx: Context, ev: TaskExecutionResultEvent
    ) -> TaskInputEvent:
        """Handle the execution result. Currently it just returns InputEvent."""
        logger.debug("📊 Handling execution result...")
        # Get the output from the event
        output = ev.output
        if output is None:
            output = "Code executed, but produced no output."
            logger.warning("  - Execution produced no output.")
        else:
            logger.debug(
                f"  - Execution output: {output[:100]}..."
                if len(output) > 100
                else f"  - Execution output: {output}"
            )
        
        # Build context-rich user message
        context_parts = []
        
        # Always remind the task
        if self.current_task:
            context_parts.append(f"Task: {self.current_task}")
        
        # Add fast incremental progress summary
        progress_context = self._get_progress_context()
        if progress_context:
            context_parts.append(progress_context)
        
        # Add the current execution result
        context_parts.append(f"Last action result:\n```\n{output}\n```")
        
        # Add guidance
        context_parts.append("What's your next action to complete the task?")
        
        # Combine all parts
        observation_message = ChatMessage(
            role="user", 
            content="\n\n".join(context_parts)
        )
        
        await self.chat_memory.aput(observation_message)

        return TaskInputEvent(input=self.chat_memory.get_all())

    @step
    async def finalize(self, ev: TaskEndEvent, ctx: Context) -> StopEvent:
        """Finalize the workflow."""
        self.tools.finished = False
        await ctx.set("chat_memory", self.chat_memory)

        # Add final state observation to episodic memory
        await self._add_final_state_observation(ctx)

        result = {}
        result.update(
            {
                "success": ev.success,
                "reason": ev.reason,
                "output": ev.reason,
                "codeact_steps": self.steps_counter,
                "code_executions": self.code_exec_counter,
            }
        )

        ctx.write_event_to_stream(EpisodicMemoryEvent(episodic_memory=self.episodic_memory))

        return StopEvent(result)

    def _format_message_for_logging(self, msg: ChatMessage) -> str:
        """Format a message for clean logging display."""
        # Check if message has image blocks
        has_screenshot = False
        text_content = ""
        
        if hasattr(msg, 'blocks') and msg.blocks:
            from llama_index.core.base.llms.types import ImageBlock
            for block in msg.blocks:
                if isinstance(block, ImageBlock):
                    has_screenshot = True
                else:
                    text_content += getattr(block, 'text', str(block)) + " "
        else:
            text_content = str(msg.content) if msg.content else ""
        
        # Clean up and truncate text
        text_content = text_content.strip()
        if len(text_content) > 300:
            text_content = text_content[:300] + "..."
        
        # Add screenshot indicator
        if has_screenshot:
            text_content = f"[📸 SCREENSHOT INCLUDED] {text_content}"
        
        return text_content

    async def _get_llm_response(
        self, ctx: Context, chat_history: List[ChatMessage], debug: bool = False
    ) -> ChatResponse | None:
        logger.debug("🔍 Getting LLM response...")
        
        # Check what's actually in the chat history
        logger.info(f"🔍 DEBUG: Chat history has {len(chat_history)} messages")
        for i, msg in enumerate(chat_history):
            if hasattr(msg, 'blocks') and msg.blocks:
                for j, block in enumerate(msg.blocks):
                    if hasattr(block, 'text') and "Current Clickable UI" in block.text:
                        logger.info(f"🔍 ✅ FOUND UI STATE in message {i}, block {j}")
                        ui_preview = block.text[:200].replace('\n', ' ')
                        logger.info(f"🔍 UI STATE PREVIEW: {ui_preview}...")

        messages_to_send = [self.system_prompt] + chat_history

        messages_to_send = chat_utils.normalize_conversation(
            messages=messages_to_send, max_tokens=4000
        )

        messages_to_send = [chat_utils.message_copy(msg) for msg in messages_to_send]

        # CLEAN CONVERSATION LOGGING - Always show what we're saying to LLM
        # Skip logging if this is a fine-tuned model (it has its own logging)
        llm_name = getattr(self.llm, '__class__', type(self.llm)).__name__
        if "FineTuned" not in llm_name:
            print("\n" + "="*60)
            print(f"🤖 CONVERSATION WITH LLM - STEP {self.steps_counter}")
            print("="*60)
        
        if "FineTuned" not in llm_name:
            for i, msg in enumerate(messages_to_send):
                role_icon = "🤖" if msg.role == "system" else "👤" if msg.role == "user" else "🤖"
                formatted_content = self._format_message_for_logging(msg)
                print(f"{role_icon} {msg.role.upper()}: {formatted_content}")
            
            print("-" * 60)
            print("⏳ Waiting for LLM response...")

        # DEBUG: Check for role alternation issues
        if debug:
            for i in range(1, len(messages_to_send)):
                if messages_to_send[i].role == messages_to_send[i - 1].role:
                    print(
                        f"❌ ROLE ISSUE: Messages {i - 1} and {i} both have role '{messages_to_send[i].role}'"
                    )

        try:
            response = await self.llm.achat(messages=messages_to_send)
            
            # CLEAN RESPONSE LOGGING - Show what LLM said back (skip for fine-tuned models)
            if "FineTuned" not in llm_name:
                response_content = response.message.content if response.message.content else ""
                if len(response_content) > 500:
                    displayed_content = response_content[:500] + "..."
                else:
                    displayed_content = response_content
                    
                print(f"🤖 ASSISTANT RESPONSE: {displayed_content}")
                print("="*60 + "\n")
            
            logger.debug("🔍 Received LLM response.")

            filtered_chat_history = []
            for msg in chat_history:
                filtered_msg = chat_utils.message_copy(msg)
                if hasattr(filtered_msg, "blocks") and filtered_msg.blocks:
                    filtered_msg.blocks = [
                        block
                        for block in filtered_msg.blocks
                        if not isinstance(block, chat_utils.ImageBlock)
                    ]
                filtered_chat_history.append(filtered_msg)

            # Convert chat history and response to JSON strings
            chat_history_str = json.dumps(
                [{"role": msg.role, "content": msg.content} for msg in filtered_chat_history]
            )
            response_str = json.dumps(
                {"role": response.message.role, "content": response.message.content}
            )

            step = EpisodicMemoryStep(
                chat_history=chat_history_str,
                response=response_str,
                timestamp=time.time(),
                screenshot=(await ctx.get("screenshot", None)),
            )

            self.episodic_memory.steps.append(step)

            assert hasattr(response, "message"), (
                f"LLM response does not have a message attribute.\nResponse: {response}"
            )
        except Exception as e:
            if self.llm.class_name() == "Gemini_LLM" and "You exceeded your current quota" in str(
                e
            ):
                s = str(e._details[2])
                match = re.search(r"seconds:\s*(\d+)", s)
                if match:
                    seconds = int(match.group(1)) + 1
                    logger.error(f"Rate limit error. Retrying in {seconds} seconds...")
                    time.sleep(seconds)
                else:
                    logger.error(f"Rate limit error. Retrying in 5 seconds...")
                    time.sleep(40)
                logger.debug("🔍 Retrying call to LLM...")
                response = await self.llm.achat(messages=messages_to_send)
            else:
                logger.error(f"Could not get an answer from LLM: {repr(e)}")
                raise e
        logger.debug("  - Received response from LLM.")
        return response

    async def _add_final_state_observation(self, ctx: Context) -> None:
        """Add the current UI state and screenshot as the final observation step."""
        try:
            # Get current screenshot and UI state
            screenshot = None
            ui_state = None

            try:
                _, screenshot_bytes = await self.tools.take_annotated_screenshot()
                screenshot = screenshot_bytes
            except Exception as e:
                logger.warning(f"Failed to capture final screenshot: {e}")

            try:
                (a11y_tree, phone_state) = await self.tools.get_state()
            except Exception as e:
                logger.warning(f"Failed to capture final UI state: {e}")

            # Create final observation chat history and response
            final_chat_history = [
                {"role": "system", "content": "Final state observation after task completion"}
            ]
            final_response = {
                "role": "user",
                "content": f"Final State Observation:\nUI State: {a11y_tree}\nScreenshot: {'Available' if screenshot else 'Not available'}",
            }

            # Create final episodic memory step
            final_step = EpisodicMemoryStep(
                chat_history=json.dumps(final_chat_history),
                response=json.dumps(final_response),
                timestamp=time.time(),
                screenshot=screenshot,
            )

            self.episodic_memory.steps.append(final_step)
            logger.info("Added final state observation to episodic memory")

        except Exception as e:
            logger.error(f"Failed to add final state observation: {e}")
