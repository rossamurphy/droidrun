import logging
from typing import List, Dict, Any, Callable
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, step
from droidrun.agent.codeact.codeact_agent import CodeActAgent
from droidrun.agent.codeact.events import TaskInputEvent, TaskThinkingEvent, TaskEndEvent
from droidrun.agent.common.events import ScreenshotEvent
from droidrun.agent.utils import chat_utils
from droidrun.agent.context.agent_persona import AgentPersona

logger = logging.getLogger("droidrun")


class VisionCodeActAgent(CodeActAgent):
    """
    A vision-focused agent that uses annotated screenshots with numbered bounding boxes
    instead of text-based UI descriptions for more efficient inference.

    Key differences from CodeActAgent:
    - Uses annotated screenshots instead of text UI descriptions
    - Shorter context window for faster inference
    - Vision-focused logging and debugging
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
        # Initialize parent class
        super().__init__(
            llm=llm,
            persona=persona,
            vision=vision,
            tools_instance=tools_instance,
            all_tools_list=all_tools_list,
            max_steps=max_steps,
            debug=debug,
            *args,
            **kwargs,
        )

        logger.info("✅ VisionCodeActAgent initialized successfully.")

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: TaskInputEvent
    ) -> TaskThinkingEvent | TaskEndEvent:
        """Handle LLM input with vision-focused approach."""
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
        logger.info(f"🧠 Step {self.steps_counter}: Vision Analysis...")

        model = self.llm.class_name()

        if "remember" in self.tool_list and self.remembered_info:
            await ctx.set("remembered_info", self.remembered_info)
            chat_history = await chat_utils.add_memory_block(self.remembered_info, chat_history)

        # Vision-based approach: Only use annotated screenshots, skip text UI descriptions
        for context in self.required_context:
            if model == "DeepSeek":
                logger.warning("[yellow]DeepSeek doesn't support images. Disabling screenshots[/]")

            elif self.vision == True and context == "screenshot":
                try:
                    # First get UI state to populate clickable elements cache
                    await self.tools.get_state()

                    # Take annotated screenshot with numbered bounding boxes
                    if hasattr(self.tools, "take_annotated_screenshot"):
                        screenshot = (await self.tools.take_annotated_screenshot())[1]
                        logger.info("📸 Using annotated screenshot with numbered bounding boxes")
                    else:
                        # Fallback to regular screenshot
                        screenshot = (await self.tools.take_screenshot())[1]
                        logger.warning("📸 Using regular screenshot (annotation not available)")

                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    await ctx.set("screenshot", screenshot)
                    chat_history = await chat_utils.add_screenshot_image_block(
                        screenshot, chat_history
                    )

                except Exception as e:
                    logger.error(f"Error taking annotated screenshot: {e}")
                    # Continue without screenshot

            # Skip text-based UI descriptions for vision agent
            # This is the key difference - we don't add ui_state or packages context

        # VISION-SPECIFIC LOGGING
        print("\n" + "📸"*60)
        print("👁️  VISION AGENT - SCREENSHOT ANALYSIS")
        print("📸"*60)
        
        # Use shorter context window for vision processing
        response = await self._get_llm_response(
            ctx, chat_history, debug=self.debug, max_tokens=2000
        )
        if response is None:
            return TaskEndEvent(
                success=False, reason="LLM response is None. This is a critical error."
            )

        await self.chat_memory.aput(response.message)

        code, thoughts = chat_utils.extract_code_and_thought(response.message.content)

        event = TaskThinkingEvent(thoughts=thoughts, code=code)
        ctx.write_event_to_stream(event)
        return event

    async def _get_llm_response(
        self,
        ctx: Context,
        chat_history: List[ChatMessage],
        debug: bool = False,
        max_tokens: int = 2000,
    ):
        """Override to use shorter context window for vision processing."""
        logger.debug("🔍 Getting LLM response for vision analysis...")

        messages_to_send = [self.system_prompt] + chat_history

        # Use shorter context for vision-based processing
        messages_to_send = chat_utils.normalize_conversation(
            messages=messages_to_send, max_tokens=max_tokens
        )

        messages_to_send = [chat_utils.message_copy(msg) for msg in messages_to_send]

        # DEBUG: Print the actual conversation structure (on each turn)
        if debug:
            print("=== VISION CONVERSATION DEBUG ===")
            for i, msg in enumerate(messages_to_send):
                content_preview = (
                    msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
                )
                print(f"{i}: {msg.role} - {content_preview}")
            print("===============================")

        # Call parent implementation for the actual LLM call
        return await super()._get_llm_response(ctx, chat_history, debug)

    async def _add_final_state_observation(self, ctx: Context) -> None:
        """Add the current UI state and screenshot as the final observation step."""
        try:
            # Get current screenshot (preferably annotated)
            screenshot = None

            try:
                if hasattr(self.tools, "take_annotated_screenshot"):
                    _, screenshot_bytes = await self.tools.take_annotated_screenshot()
                else:
                    _, screenshot_bytes = await self.tools.take_screenshot()
                screenshot = screenshot_bytes
            except Exception as e:
                logger.warning(f"Failed to capture final screenshot: {e}")

            # Create final observation for episodic memory
            final_chat_history = [
                {"role": "system", "content": "Final state observation after task completion"}
            ]
            final_response = {
                "role": "user",
                "content": f"Final State Observation: Screenshot saved with final UI state.",
            }

            # Create final episodic memory step
            from droidrun.agent.context.episodic_memory import EpisodicMemoryStep
            import json
            import time

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

