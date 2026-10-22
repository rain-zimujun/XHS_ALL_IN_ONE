"""LangChain chat-model adapter using the application's existing transport.

No credentials enter model fields, graph state, or LangChain serialization.
Native JSON Schema is required; unsupported providers never fall back silently.
"""
from copy import deepcopy
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_json_schema
from pydantic import PrivateAttr, TypeAdapter
from typing_extensions import is_typeddict


def response_parameter(response_format: type) -> dict:
    if not is_typeddict(response_format):
        raise TypeError("response_format must be an explicit TypedDict class")
    # TypeAdapter retains Annotated bounds and nested TypedDict definitions.
    # Converting TypedDict directly in core 0.3 uses Pydantic v1 and loses them.
    schema = TypeAdapter(response_format).json_schema()
    schema = convert_to_json_schema(schema, strict=True)
    return {"type": "json_schema", "json_schema": {
        "name": response_format.__name__, "strict": True, "schema": schema,
    }}


class CompatibleChatModel(BaseChatModel):
    """A single model request per invocation, with retries owned by the workflow."""

    _transport: Callable = PrivateAttr()

    def __init__(self, complete: Callable):
        super().__init__(cache=False, callbacks=[], metadata={})
        self._transport = complete

    @property
    def _llm_type(self) -> str:
        return "creative-openai-compatible"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if stop is not None or [m.type for m in messages] != ["system", "human"]:
            raise ValueError("Creative agents require exactly a system and a user message")
        raw = self._transport(
            system_prompt=messages[0].content, user_prompt=messages[1].content,
            temperature=kwargs.get("temperature", 0.4),
            response_format=deepcopy(kwargs["response_format"]),
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=raw))])

    def with_structured_output(self, schema, *, include_raw=False, method="json_schema", strict=True, temperature=0.4, **kwargs):
        if include_raw or method != "json_schema" or strict is not True or kwargs:
            raise ValueError("Creative agents require strict native JSON Schema output")
        parameter = response_parameter(schema)
        adapter = TypeAdapter(schema)
        # JsonOutputParser accepts partial JSON, so use its strict parser entry
        # point through parse_json_object before validating the complete object.
        from backend.app.services.creative_orchestrator import parse_json_object

        def validate(message):
            value = parse_json_object(message.content)
            return adapter.validate_python(value, strict=True)

        return self.bind(response_format=parameter, temperature=temperature) | RunnableLambda(validate)
