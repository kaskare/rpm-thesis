"""Model backends. Each returns a Reply with the visible text and metadata.

anthropic : Claude via the Anthropic SDK (ANTHROPIC_API_KEY).
openai    : OpenAI Chat Completions (OPENAI_API_KEY). With --base-url it also covers any
            OpenAI-compatible server: vLLM / Ollama / LM Studio for Gemma and Qwen,
            Mistral (https://api.mistral.ai/v1), Gemini's OpenAI endpoint, OpenRouter.
"""
import base64
import os
from dataclasses import dataclass, field


@dataclass
class Reply:
    text: str
    stop_reason: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)


class AnthropicProvider:
    def __init__(self, model, max_tokens=32000, effort=None, temperature=None, **_):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=6)
        self.model, self.max_tokens, self.effort, self.temperature = model, max_tokens, effort, temperature

    def __call__(self, prompt, png):
        kw = {}
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        if self.temperature is not None:  # rejected (400) by Claude 4.7+ models
            kw["temperature"] = self.temperature
        content = [{"type": "text", "text": prompt}]
        if png is not None:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                           "data": base64.standard_b64encode(png).decode()}})
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens,
                                         messages=[{"role": "user", "content": content}], **kw) as s:
            msg = s.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Reply(text, msg.stop_reason or "", msg.model, msg.usage.to_dict())


class OpenAIProvider:
    def __init__(self, model, max_tokens=32000, effort=None, temperature=None, base_url=None,
                 api_key_env="OPENAI_API_KEY", **_):
        import openai
        key = os.environ.get(api_key_env) or ("EMPTY" if base_url else None)
        self.client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=6, timeout=1800)
        self.model, self.max_tokens, self.effort, self.temperature = model, max_tokens, effort, temperature
        self.base_url = base_url

    def __call__(self, prompt, png):
        kw = {}
        if self.effort:
            kw["reasoning_effort"] = self.effort
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        # Official OpenAI reasoning models only accept max_completion_tokens; most local servers take max_tokens.
        kw["max_tokens" if self.base_url else "max_completion_tokens"] = self.max_tokens
        content = [{"type": "text", "text": prompt}]
        if png is not None:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], **kw)
        ch = r.choices[0]
        usage = r.usage.model_dump() if r.usage else {}
        return Reply(ch.message.content or "", ch.finish_reason or "", r.model, usage)


PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}
