from typing import Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.workflow import Event

from droidrun.agent.context import Task


class PlanInputEvent(Event):
    input: list[ChatMessage]


class PlanThinkingEvent(Event):
    thoughts: Optional[str] = None
    code: Optional[str] = None


class PlanCreatedEvent(Event):
    tasks: list[Task]
