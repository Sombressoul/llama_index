from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from llama_index.bridge.pydantic import (
    BaseModel as PydanticBaseModel,
    Field,
)
from llama_index.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.llm import LLM
from llama_index.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

"""
To implement:
# --------------------------------------------------------------------------------------------------------------- #

@property
def _model_kwargs(self) -> Dict[str, Any]:

def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:

# --------------------------------------------------------------------------------------------------------------- #

@llm_chat_callback()
def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:

@llm_chat_callback()
def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:

@llm_chat_callback()
async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:

@llm_chat_callback()
async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:

# --------------------------------------------------------------------------------------------------------------- #

@llm_completion_callback()
def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:

@llm_completion_callback()
def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:

@llm_completion_callback()
async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:

@llm_completion_callback()
async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:

# --------------------------------------------------------------------------------------------------------------- #
"""

VLLM_VERSION = "0.2.6"

try:
    import torch

    LogitsProcessor = Callable[[List[int], torch.Tensor], torch.Tensor]
except ImportError:
    TensorLike = Union[List[int], List[float]]  # simple replacement for torch.Tensor
    LogitsProcessor = Callable[[List[int], TensorLike], TensorLike]


def kwargs_only(f):
    """Decorator to make a function only accept kwargs."""

    def f_decorated(**kwargs):
        return f(**kwargs)

    return f_decorated


class VllmSamplingParams(PydanticBaseModel):
    """Sampling parameters for text generation.

    Please, refer to the implementation of SamplingParams from the original vLLM repository: \
    [SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)
    """

    n: int = Field(
        default=1,
        description="Number of output sequences to return for the given prompt.",
        gte=0,
    )

    best_of: int = Field(
        default=1,
        description=(
            "Number of output sequences that are generated from the prompt."
            "From these `best_of` sequences, the top `n` sequences are returned."
            "`best_of` must be greater than or equal to `n`. This is treated as"
            "the beam width when `use_beam_search` is True. By default, `best_of`"
            "is set to `n`."
        ),
        gte=0,
    )

    presence_penalty: float = Field(
        default=0.0,
        description=(
            "Float that penalizes new tokens based on whether they"
            "appear in the generated text so far. Values > 0 encourage the model"
            "to use new tokens, while values < 0 encourage the model to repeat"
            "tokens."
        ),
    )

    frequency_penalty: float = Field(
        default=0.0,
        description=(
            "Float that penalizes new tokens based on their"
            "frequency in the generated text so far. Values > 0 encourage the"
            "model to use new tokens, while values < 0 encourage the model to"
            "repeat tokens."
        ),
    )

    repetition_penalty: float = Field(
        default=1.0,
        description=(
            "Float that penalizes new tokens based on whether"
            "they appear in the prompt and the generated text so far. Values > 1"
            "encourage the model to use new tokens, while values < 1 encourage"
            "the model to repeat tokens."
        ),
        ge=0,
    )

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description=(
            "Float that controls the randomness of the sampling. Lower"
            "values make the model more deterministic, while higher values make"
            "the model more random. Zero means greedy sampling."
        ),
        ge=0,
    )

    top_p: float = Field(
        default=1.0,
        description=(
            "Float that controls the cumulative probability of the top tokens"
            "to consider. Must be in (0, 1]. Set to 1 to consider all tokens."
        ),
        gt=0,
        le=1,
    )

    top_k: int = Field(
        default=-1,
        description=(
            "Integer that controls the number of top tokens to consider. Set"
            "to -1 to consider all tokens."
        ),
        ge=-1,
    )

    min_p: float = Field(
        default=0.0,
        description=(
            "Float that represents the minimum probability for a token to be"
            "considered, relative to the probability of the most likely token."
            "Must be in [0, 1]. Set to 0 to disable this."
        ),
        ge=0,
        le=1,
    )

    use_beam_search: bool = Field(
        default=False,
        description="Whether to use beam search instead of sampling.",
    )

    length_penalty: float = Field(
        default=1.0,
        description=(
            "Float that penalizes sequences based on their length."
            "Used in beam search."
        ),
    )

    early_stopping: Union[bool, str] = Field(
        default=False,
        description=(
            "Controls the stopping condition for beam search. It"
            "accepts the following values: `True`, where the generation stops as"
            "soon as there are `best_of` complete candidates; `False`, where an"
            "heuristic is applied and the generation stops when is it very"
            'unlikely to find better candidates; `"never"`, where the beam search'
            "procedure only stops when there cannot be better candidates"
            "(canonical beam search algorithm)."
        ),
    )

    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "List of strings that stop the generation when they are generated."
            "The returned output will not contain the stop strings."
        ),
    )

    stop_token_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "List of tokens that stop the generation when they are"
            "generated. The returned output will contain the stop tokens unless"
            "the stop tokens are special tokens."
        ),
    )

    include_stop_str_in_output: bool = Field(
        default=False,
        description="Whether to include the stop strings in output text. Defaults to False.",
    )

    ignore_eos: bool = Field(
        default=False,
        description=(
            "Whether to ignore the EOS token and continue generating"
            "tokens after the EOS token is generated."
        ),
    )

    max_tokens: int = Field(
        default=16,
        description="Maximum number of tokens to generate per output sequence.",
        gt=0,
    )

    logprobs: Optional[int] = Field(
        default=None,
        description=(
            "Number of log probabilities to return per output token."
            "Note that the implementation follows the OpenAI API: The return"
            "result includes the log probabilities on the `logprobs` most likely"
            "tokens, as well the chosen tokens. The API will always return the"
            "log probability of the sampled token, so there  may be up to"
            "`logprobs+1` elements in the response."
        ),
    )

    prompt_logprobs: Optional[int] = Field(
        default=None,
        description="Number of log probabilities to return per prompt token.",
    )

    skip_special_tokens: bool = Field(
        default=True,
        description="Whether to skip special tokens in the output.",
    )

    spaces_between_special_tokens: bool = Field(
        default=True,
        description=(
            "Whether to add spaces between special"
            "tokens in the output. Defaults to True."
        ),
    )

    logits_processors: Optional[List[LogitsProcessor]] = Field(
        default=None,
        description=(
            "List of functions that modify logits based on"
            "previously generated tokens."
        ),
    )

    @kwargs_only
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        pass


class VllmBase:
    """Base class for both vLLM sever and client modules."""

    model: Optional[str] = Field(
        default=None,
        description="Model name/id to call on server side.",
    )

    legacy_mode: bool = Field(
        default=True,
        description=(
            "If true, the llama-index will use the legacy vLLM module"
            "implementation (llama-index<=0.9.23)."
        ),
    )

    @kwargs_only
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Basic initialization for vLLM.

        Args:

        Returns:
            None

        Raises:
            Deprecation warning if legacy_mode is true.
        """
        if self.legacy_mode:
            raise DeprecationWarning(
                "Legacy mode is deprecated and will be removed in a future release. Please migrate to the new API."
            )
        pass


class VllmServer(VllmBase):
    ...


class VllmServerOpenAI(VllmServer):
    ...


class VllmClient(LLM, VllmSamplingParams, VllmBase):
    """Client class for vLLM."""

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )

    is_chat_model: bool = Field(
        default=False,
        description="Whether the model is a chat model.",
    )

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=self.is_chat_model,
        )

    @property
    def _model_kwargs_sampling(
        self,
    ) -> Dict[str, Any]:
        keys = VllmSamplingParams.__fields__.keys()
        vals = [getattr(self, k) for k in keys]
        return dict(zip(keys, vals))

    @property
    def _model_kwargs(
        self,
    ) -> Dict[str, Any]:
        ...  # TODO: implementation.

    @classmethod
    def class_name(cls) -> str:
        return "vllm_client"

    @kwargs_only
    def __init__(
        self,
        legacy_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(legacy_mode, **kwargs)
        pass

    def _complete(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        ...  # TODO: implementation.
        return CompletionResponse(...)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        ...  # TODO: implementation.

        def gen() -> CompletionResponseGen:
            ...  # TODO: implementation.

        return gen()

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        ...  # TODO: implementation.
        return CompletionResponse(...)

    # TODO: BACKBONE: astream_chat, astream_complete, chat, complete, stream_chat, stream_complete


class VllmClientOpenAI(VllmClient):
    # vLLM OpenAI API server supports chat API for all models.
    is_chat_model: bool = True

    def __init__(
        self,
        legacy_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(legacy_mode, **kwargs)
        pass

    @classmethod
    def class_name(cls) -> str:
        return "vllm_client_openai"

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        ...  # TODO: implementation.
        return ChatResponse(...)
