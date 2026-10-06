"""Thin provider layer for the API-model calls (dilemma generation, behavior judging).

Provider and model come from flags or the environment (.env is loaded automatically):
  MFT_PROVIDER  openai (default) | anthropic | bedrock
  BEDROCK_MODEL  defaults to deepseek.v3.2 (any Bedrock Converse model id); uses AWS credentials
  OPENAI_MODEL  required when the provider is openai
  ANTHROPIC_MODEL  defaults to claude-opus-5-5
Both helpers return None when the model refuses or returns unparseable output.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_BEDROCK_MODEL = "deepseek.v3.2"


@dataclass
class LLM:
    provider: str
    model: str
    client: object

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> "LLM":
        provider = provider or os.environ.get("MFT_PROVIDER", "openai")
        if provider == "openai":
            import openai

            model = model or os.environ.get("OPENAI_MODEL")
            if not model:
                raise SystemExit("Set OPENAI_MODEL in .env or pass --llm-model for the openai provider.")
            return cls(provider, model, openai.OpenAI(max_retries=6))
        if provider == "anthropic":
            import anthropic

            return cls(provider, model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
                       anthropic.Anthropic(max_retries=6))
        if provider == "bedrock":
            import boto3
            from botocore.config import Config

            client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                                  config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=600))
            return cls(provider, model or os.environ.get("BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL), client)
        raise SystemExit(f"Unknown provider {provider!r}; use openai, anthropic or bedrock.")

    def structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
        if self.provider == "bedrock":
            return self._bedrock_structured(system, user, schema)
        if self.provider == "openai":
            response = self.client.responses.parse(
                model=self.model, instructions=system, input=user, text_format=schema,
            )
            return response.output_parsed  # None on refusal

        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": strict_schema(schema)}},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            return None
        text = next(b.text for b in response.content if b.type == "text")
        try:
            return schema.model_validate_json(text)
        except ValidationError:
            return None

    def text(self, prompt: str) -> str | None:
        if self.provider == "bedrock":
            return self._bedrock_converse(None, prompt, max_tokens=2000)
        if self.provider == "openai":
            return self.client.responses.create(model=self.model, input=prompt).output_text or None

        response = self.client.beta.messages.create(
            model=self.model, max_tokens=2000, output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), None)

    def _bedrock_converse(self, system: str | None, user: str, max_tokens: int = 16000) -> str | None:
        kwargs = {"system": [{"text": system}]} if system else {}
        response = self.client.converse(
            modelId=self.model, messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens}, **kwargs,
        )
        if response["stopReason"] in ("content_filtered", "guardrail_intervened"):
            return None
        return "".join(b.get("text", "") for b in response["output"]["message"]["content"]) or None

    def _bedrock_structured(self, system: str, user: str, schema: type[BaseModel], attempts: int = 3) -> BaseModel | None:
        """No portable constrained decoding on Bedrock: ask for JSON matching the schema, validate, retry."""
        instruction = (
            "\n\nReturn only a JSON object (no prose, no code fences) that validates against this JSON schema:\n"
            + json.dumps(strict_schema(schema))
        )
        for _ in range(attempts):
            text = self._bedrock_converse(system, user + instruction)
            if text is None:
                return None
            start, end = text.find("{"), text.rfind("}")
            try:
                return schema.model_validate_json(text[start : end + 1])
            except ValidationError:
                continue
        return None

    @property
    def api_errors(self) -> tuple[type[Exception], ...]:
        if self.provider == "bedrock":
            from botocore.exceptions import BotoCoreError, ClientError

            return (BotoCoreError, ClientError)
        if self.provider == "openai":
            import openai

            return (openai.APIError,)
        import anthropic

        return (anthropic.APIError,)


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with additionalProperties: false and all fields required, as strict
    structured outputs require."""
    schema = json.loads(json.dumps(model.model_json_schema()))

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(schema)
    return schema
