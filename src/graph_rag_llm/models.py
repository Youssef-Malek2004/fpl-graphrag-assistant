# src/graph_rag_llm/models.py
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI


class LLMBackend:
    """
    Minimal backend: ONLY OpenRouter model, no dummy, no OpenAI standard.
    """

    # src/graph_rag_llm/models.py
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,  # ✅ use the passed key
        )

    def generate(
            self,
            model: str,
            messages: List[Dict[str, str]],
            max_tokens: int = 512,
    ) -> Tuple[str, Dict[str, Any]]:
        start = time.time()

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        text = completion.choices[0].message.content
        usage = completion.usage

        meta = {
            "latency": time.time() - start,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        return text, meta
