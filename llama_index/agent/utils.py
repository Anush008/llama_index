"""Agent utils."""


from typing import Any, Optional

from llama_index.agent.runner.base import AgentRunner
from llama_index.agent.types import TaskStep
from llama_index.core.llms.types import MessageRole
from llama_index.llms.base import ChatMessage
from llama_index.llms.llm import LLM
from llama_index.memory import BaseMemory


def add_user_step_to_memory(
    step: TaskStep, memory: BaseMemory, verbose: bool = False
) -> None:
    """Add user step to memory."""
    user_message = ChatMessage(content=step.input, role=MessageRole.USER)
    memory.put(user_message)
    if verbose:
        print(f"Added user message to memory: {step.input}")


def create_agent_from_llm(
    llm: Optional[LLM] = None,
    **kwargs: Any,
) -> AgentRunner:
    from llama_index.agent import OpenAIAgent, ReActAgent
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai_utils import is_function_calling_model

    if isinstance(llm, OpenAI) and is_function_calling_model(llm.model):
        return OpenAIAgent.from_tools(
            llm=llm,
            **kwargs,
        )
    else:
        return ReActAgent.from_tools(
            llm=llm,
            **kwargs,
        )
