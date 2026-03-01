import os
import json
from typing import List
import boto3
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_random_exponential
from kernelband.models.base import BaseModel


class ClaudeModel(BaseModel):
    def __init__(self,
                 model_id="claude-sonnet-4-5-20250929-v1:0",
                 api_key=None,
                 base_url=None,
                 region_name="us-east-1"):
        """
        Initialize Claude model client via AWS Bedrock.

        Args:
            model_id: Model ID or inference profile ID
                     (e.g., 'claude-sonnet-4-5-20250929-v1:0' will be converted to
                      'global.anthropic.claude-sonnet-4-5-20250929-v1:0')
            api_key: Not used for AWS Bedrock (kept for interface compatibility)
            base_url: Not used for AWS Bedrock (kept for interface compatibility)
            region_name: AWS region (default: us-east-1, can be set via AWS_REGION env var)
        """
        if 'AWS_REGION' in os.environ:
            region_name = os.environ['AWS_REGION']

        if not model_id.startswith("us.") and not model_id.startswith("arn:"):
            # Handle both full model IDs and short names
            if model_id.startswith("anthropic."):
                self.model_id = f"us.{model_id}"
            else:
                self.model_id = f"us.anthropic.{model_id}"
        else:
            self.model_id = model_id

        # boto3 will automatically use AWS credentials from:
        # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
        # 2. AWS credentials file (~/.aws/credentials)
        # 3. IAM role (if running on EC2/ECS/Lambda)
        # Configure timeout for slow models (some need 800+ seconds)
        bedrock_config = Config(
            read_timeout=1200,      # 20 minutes read timeout for slow inference
            connect_timeout=60,     # 60 seconds connection timeout
            retries={'max_attempts': 0}  # Disable SDK retries (use tenacity @retry)
        )
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            config=bedrock_config
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self,
                 messages: List,
                 temperature=0,
                 presence_penalty=0,
                 frequency_penalty=0,
                 max_tokens=50000,
                 max_completion_tokens=50000
                 ) -> str:
        """
        Generate response using AWS Bedrock Claude API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            presence_penalty: Not supported by AWS Bedrock (ignored)
            frequency_penalty: Not supported by AWS Bedrock (ignored)
            max_tokens: Maximum tokens to generate
            max_completion_tokens: Alternative name for max_tokens

        Returns:
            Generated text response as string
        """
        # Use max_completion_tokens if it's different from default, otherwise use max_tokens
        actual_max_tokens = max_completion_tokens if max_completion_tokens != 50000 else max_tokens

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": actual_max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())

        parts = response_body.get("content", [])
        text_chunks = [
            p.get("text", "") for p in parts
            if p.get("type") == "text"
        ]

        return "".join(text_chunks)
    
