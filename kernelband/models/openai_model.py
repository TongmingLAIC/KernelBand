import os
from typing import List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from kernelband.models.base import BaseModel


class OpenAIModel(BaseModel):
    def __init__(self,
                 model_id="gpt-4o",
                 api_key=None,
                 base_url=None,
                 use_responses_api=False):
        """
        Initialize OpenAI model client.

        Args:
            model_id: Model name (e.g., 'gpt-4o', 'o3', 'o4-mini')
            api_key: OpenAI API key (if None, will try to read from OPENAI_API_KEY env var)
            base_url: Optional custom endpoint (None for direct OpenAI platform)
            use_responses_api: Whether to use new Responses API by default
        """
        if not api_key:  # This handles both None and empty string
            api_key = os.environ.get('OPENAI_API_KEY')

        assert api_key, "No API key provided. Please set it in config or OPENAI_API_KEY environment variable."
        self.model_id = model_id
        self.use_responses_api = use_responses_api

        if not base_url:  # This handles both None and empty string
            if 'MODEL_API_URL' in os.environ:
                base_url = os.environ['MODEL_API_URL']

        # Initialize standard OpenAI client
        # timeout: 20 minutes for slow models (some models need 800+ seconds)
        # max_retries: 0 to avoid double-retry (tenacity handles retries)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,  # None for direct OpenAI platform, or custom endpoint
            timeout=1200.0,     # 20 minutes timeout for slow inference
            max_retries=0       # Disable SDK retries (use tenacity @retry instead)
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self,
                 messages: List,
                 temperature=0,
                 presence_penalty=0,
                 frequency_penalty=0,
                 max_tokens=5000,
                 reasoning_effort=None,
                 reasoning_summary=None) -> str:
        """
        Generate a response using OpenAI API.
        Automatically chooses between Responses API and Chat Completions API based on model and parameters.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            max_tokens: Maximum output tokens
            reasoning_effort: For reasoning models - effort level ('low', 'medium', 'high')
            reasoning_summary: For reasoning models - summary type ('auto', 'concise', 'detailed')

        Returns:
            Generated text response as string
        """
        is_reasoning_model = (
            'o3' in self.model_id.lower() or
            'o4' in self.model_id.lower() or
            reasoning_effort is not None or
            reasoning_summary is not None
        )

        use_responses_api = is_reasoning_model or self.use_responses_api

        if use_responses_api:
            instructions = None
            input_text = None

            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')

                if role in ['system', 'developer']:
                    instructions = content if instructions is None else f"{instructions}\n{content}"
                elif role == 'user':
                    input_text = content if input_text is None else f"{input_text}\n{content}"
                elif role == 'assistant' and input_text is None:
                    # If assistant message appears before user message, treat as context
                    instructions = f"{instructions}\n{content}" if instructions else content

            params = {
                "model": self.model_id,
                "instructions": instructions,
                "input": input_text or "",
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if is_reasoning_model and (reasoning_effort or reasoning_summary):
                reasoning_config = {}
                if reasoning_effort:
                    reasoning_config["effort"] = reasoning_effort
                if reasoning_summary:
                    reasoning_config["summary"] = reasoning_summary
                params["reasoning"] = reasoning_config

            try:
                response = self.client.responses.create(**params)

                if not response or not hasattr(response, 'output_text'):
                    raise ValueError("No output_text returned from Responses API.")

                return response.output_text

            except Exception as e:
                # If Responses API fails and it's not a reasoning-only model, fallback to Chat Completions
                if not is_reasoning_model:
                    print(f"Responses API failed: {e}. Falling back to Chat Completions.")
                    use_responses_api = False
                else:
                    # For reasoning models, we must use Responses API
                    raise

        if not use_responses_api:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                n=1,
                stream=False,
                stop=None,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=None,
                user=None
            )
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                raise ValueError("No response choices returned from Chat Completions API.")

            return response.choices[0].message.content
    