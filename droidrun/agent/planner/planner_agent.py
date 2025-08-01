import asyncio
import logging
from typing import TYPE_CHECKING, List, Union

from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import Memory
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from droidrun.agent.common.events import ScreenshotEvent
from droidrun.agent.context.agent_persona import AgentPersona
from droidrun.agent.context.reflection import Reflection
from droidrun.agent.context.task_manager import TaskManager
from droidrun.agent.planner.events import *
from droidrun.agent.planner.events import (
    PlanCreatedEvent,
    PlanInputEvent,
    PlanThinkingEvent,
)
from droidrun.agent.planner.prompts import (
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    DEFAULT_PLANNER_USER_PROMPT,
)
from droidrun.agent.utils import chat_utils
from droidrun.agent.utils.executer import SimpleCodeExecutor
from droidrun.tools import Tools

load_dotenv()

# Setup logger
logger = logging.getLogger("droidrun")

if TYPE_CHECKING:
    from droidrun.tools import Tools


class PlannerAgent(Workflow):
    def __init__(
        self,
        goal: str,
        llm: LLM,
        vision: bool,
        personas: List[AgentPersona],
        task_manager: TaskManager,
        tools_instance: Tools,
        system_prompt=None,
        user_prompt=None,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.goal = goal
        self.task_manager = task_manager
        self.debug = debug
        self.vision = vision

        self.chat_memory = None
        self.remembered_info = None
        self.reflection: Reflection = None

        self.current_retry = 0
        self.steps_counter = 0

        self.tool_list = {}
        self.tool_list[self.task_manager.set_tasks_with_agents.__name__] = (
            self.task_manager.set_tasks_with_agents
        )
        self.tool_list[self.task_manager.complete_goal.__name__] = self.task_manager.complete_goal

        self.tools_description = chat_utils.parse_tool_descriptions(self.tool_list)
        self.tools_instance = tools_instance

        self.personas = personas

        self.system_prompt = system_prompt or DEFAULT_PLANNER_SYSTEM_PROMPT.format(
            tools_description=self.tools_description,
            agents=chat_utils.parse_persona_description(self.personas),
        )
        self.user_prompt = user_prompt or DEFAULT_PLANNER_USER_PROMPT.format(goal=goal)
        self.system_message = ChatMessage(
            role="system",
            content=self.system_prompt,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )
        self.user_message = ChatMessage(
            role="user",
            content=self.user_prompt,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )

        self.executer = SimpleCodeExecutor(
            loop=asyncio.get_event_loop(), globals={}, locals={}, tools=self.tool_list
        )

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> PlanInputEvent:
        logger.info("💬 Preparing planning session...")

        self.chat_memory: Memory = await ctx.get("chat_memory", default=Memory.from_defaults())
        await self.chat_memory.aput(self.user_message)

        if ev.remembered_info:
            self.remembered_info = ev.remembered_info

        if ev.reflection:
            self.reflection = ev.reflection
        else:
            self.reflection = None

        assert len(self.chat_memory.get_all()) > 0 or self.user_prompt, (
            "Memory input, user prompt or user input cannot be empty."
        )

        await self.chat_memory.aput(
            ChatMessage(
                role="user",
                content=PromptTemplate(
                    self.user_prompt or DEFAULT_PLANNER_USER_PROMPT.format(goal=self.goal)
                ),
            )
        )

        input_messages = self.chat_memory.get_all()
        logger.debug(f"  - Memory contains {len(input_messages)} messages")
        return PlanInputEvent(input=input_messages)

    @step
    async def handle_llm_input(self, ev: PlanInputEvent, ctx: Context) -> PlanThinkingEvent:
        """Handle LLM input."""
        chat_history = ev.input
        assert len(chat_history) > 0, "Chat history cannot be empty."

        ctx.write_event_to_stream(ev)

        self.steps_counter += 1
        logger.info("🧠 Thinking about how to plan the goal...")

        if self.vision:
            screenshot = (await self.tools_instance.take_screenshot())[1]
            ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
            await ctx.set("screenshot", screenshot)

        try:
            state = await self.tools_instance.get_state()
            await ctx.set("ui_state", state["a11y_tree"])
            await ctx.set("phone_state", state["phone_state"])
        except Exception as e:
            logger.warning(f"Exception Raised: {e}")
            logger.warning(
                "⚠️ Error retrieving state from the connected device. Is the Accessibility Service enabled?"
            )

        await ctx.set("remembered_info", self.remembered_info)
        await ctx.set("reflection", self.reflection)

        response = await self._get_llm_response(ctx, chat_history)
        await self.chat_memory.aput(response.message)

        code, thoughts = chat_utils.extract_code_and_thought(response.message.content)

        event = PlanThinkingEvent(thoughts=thoughts, code=code)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_llm_output(
        self, ev: PlanThinkingEvent, ctx: Context
    ) -> Union[PlanInputEvent, PlanCreatedEvent]:
        """Handle LLM output."""
        logger.debug("🤖 Processing planning output...")
        code = ev.code
        thoughts = ev.thoughts

        if code:
            try:
                result = await self.executer.execute(ctx, code)
                logger.info("📝 Planning complete")
                logger.debug(f"  - Planning code executed. Result: {result}")

                await self.chat_memory.aput(
                    ChatMessage(role="user", content=f"Execution Result:\n```\n{result}\n```")
                )

                self.remembered_info = self.tools_instance.memory

                tasks = self.task_manager.get_all_tasks()
                event = PlanCreatedEvent(tasks=tasks)

                if not self.task_manager.goal_completed:
                    logger.info(f"📋 Current plan created with {len(tasks)} tasks:")
                    for i, task in enumerate(tasks):
                        logger.info(
                            f"  Task {i}: [{task.status.upper()}] [{task.agent_type}] {task.description}"
                        )
                    ctx.write_event_to_stream(event)

                return event

            except Exception as e:
                logger.debug(f"error handling Planner: {e}")
                await self.chat_memory.aput(
                    ChatMessage(
                        role="user",
                        content="""Please either set new tasks using set_tasks_with_agents() or mark the goal as complete using complete_goal() if done.
wrap your code inside this:
```python
<YOUR CODE HERE>
```""",
                    )
                )
                logger.debug("🔄 Waiting for next plan or completion.")
                return PlanInputEvent(input=self.chat_memory.get_all())
        else:
            await self.chat_memory.aput(
                ChatMessage(
                    role="user",
                    content="""Please either set new tasks using set_tasks_with_agents() or mark the goal as complete using complete_goal() if done.
wrap your code inside this:
```python
<YOUR CODE HERE>
```""",
                )
            )
            logger.debug("🔄 Waiting for next plan or completion.")
            return PlanInputEvent(input=self.chat_memory.get_all())

    @step
    async def finalize(self, ev: PlanCreatedEvent, ctx: Context) -> StopEvent:
        """Finalize the workflow."""
        await ctx.set("chat_memory", self.chat_memory)

        result = {}
        result.update(
            {
                "tasks": ev.tasks,
            }
        )

        return StopEvent(result=result)

    async def _get_llm_response(
        self, ctx: Context, chat_history: List[ChatMessage]
    ) -> ChatResponse:
        """Get streaming response from LLM."""
        try:
            logger.debug(f"  - Sending {len(chat_history)} messages to LLM.")

            model = self.llm.class_name()
            if model == "DeepSeek":
                logger.warning("[yellow]DeepSeek doesnt support images. Disabling screenshots[/]")

            elif self.vision == True:
                chat_history = await chat_utils.add_screenshot_image_block(
                    await ctx.get("screenshot"), chat_history
                )

            chat_history = await chat_utils.add_task_history_block(
                self.task_manager.get_completed_tasks(),
                self.task_manager.get_failed_tasks(),
                chat_history,
            )

            remembered_info = await ctx.get("remembered_info", default=None)
            if remembered_info:
                chat_history = await chat_utils.add_memory_block(remembered_info, chat_history)

            reflection = await ctx.get("reflection", None)
            if reflection:
                chat_history = await chat_utils.add_reflection_summary(reflection, chat_history)

            chat_history = await chat_utils.add_phone_state_block(
                await ctx.get("phone_state"), chat_history
            )
            chat_history = await chat_utils.add_ui_text_block(
                await ctx.get("ui_state"), chat_history
            )

            messages_to_send = [self.system_message] + chat_history
            messages_to_send = [chat_utils.message_copy(msg) for msg in messages_to_send]

            logger.debug(f"  - Final message count: {len(messages_to_send)}")

            response = await self.llm.achat(messages=messages_to_send)
            assert hasattr(response, "message"), (
                f"LLM response does not have a message attribute.\nResponse: {response}"
            )
            logger.debug("  - Received response from LLM.")
            return response
        except Exception as e:
            logger.error(f"Could not get an answer from LLM: {repr(e)}")
            raise e
