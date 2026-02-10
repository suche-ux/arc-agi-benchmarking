"""Gemini Adapter using the Interactions API

This adapter uses client.interactions.create() instead of
client.models.generate_content().
It supports background mode for models that require async polling.
See documentation: https://ai.google.dev/gemini-api/docs/interactions?ua=chat.
"""

from datetime import datetime, timezone
import json
import logging
import os
import time
from typing import List, Optional
from google import genai
from google3.experimental.users.rburnell.arc_agi.arc_agi_benchmarking.src.arc_agi_benchmarking.schemas import (
    ARCTaskOutput,
    Attempt,
    AttemptMetadata,
    Choice,
    CompletionTokensDetails,
    Cost,
    Message,
    Usage,
)
from .provider import ProviderAdapter

logger = logging.getLogger(__name__)


class _InteractionResponse:
  """Wrapper for Interaction response to provide consistent interface."""

  def __init__(self, interaction):
    self._interaction = interaction
    self._text = None
    self._usage = getattr(interaction, "usage", None)

  @property
  def text(self) -> str:
    if self._text is None:
      text_parts = []
      for output in self._interaction.outputs:
        if hasattr(output, "type") and output.type == "text":
          output_text = getattr(output, "text", "")
          if output_text:
            text_parts.append(output_text)
      if text_parts:
        self._text = "".join(text_parts)
      elif self._interaction.outputs:
        self._text = getattr(self._interaction.outputs[-1], "text", "")
      else:
        self._text = ""
    return self._text

  @property
  def usage_metadata(self):
    """Return usage in a format compatible with the base adapter."""
    return self._usage


class _StreamResponse:
  """Wrapper for streaming response to hold accumulated text and metadata."""

  def __init__(self, text: str, usage_metadata):
    self.text = text
    self.usage_metadata = usage_metadata


class GeminiInteractionsAdapter(ProviderAdapter):
  """Adapter for Gemini models using the Interactions API."""

  def init_client(self):
    """Initialize the Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
      raise ValueError("GOOGLE_API_KEY not found in environment variables")

    self.generation_config_dict = self.model_config.kwargs.copy()
    self.background_mode = self.model_config.kwargs.get(
        "background", False
    ) or getattr(self.model_config, "background", False)
    self.poll_interval = self.model_config.kwargs.get("poll_interval", 1.0)

    client = genai.Client(api_key=api_key)
    return client

  def make_prediction(
      self,
      prompt: str,
      task_id: Optional[str] = None,
      test_id: Optional[str] = None,
      pair_index: int = None,
  ) -> Attempt:
    """Make a prediction using the Interactions API."""
    start_time = datetime.now(timezone.utc)

    messages = [{"role": "user", "content": prompt}]

    stream_enabled = self.model_config.kwargs.get("stream", False) or getattr(
        self.model_config, "stream", False
    )

    if stream_enabled:
      response = self.chat_completion_stream(messages)
    else:
      response = self.chat_completion(messages)

    if response is None:
      logger.error(
          f"Failed to get response from chat_completion for task {task_id}"
      )
      default_usage = Usage(
          prompt_tokens=0,
          completion_tokens=0,
          total_tokens=0,
          completion_tokens_details=CompletionTokensDetails(
              reasoning_tokens=0,
              accepted_prediction_tokens=0,
              rejected_prediction_tokens=0,
          ),
      )
      default_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)

      return Attempt(
          metadata=AttemptMetadata(
              model=self.model_config.model_name,
              provider=self.model_config.provider,
              start_timestamp=start_time,
              end_timestamp=datetime.now(timezone.utc),
              choices=[],
              kwargs=self.model_config.kwargs,
              usage=default_usage,
              cost=default_cost,
              error_message="Failed to get valid response from provider",
              task_id=task_id,
              pair_index=pair_index,
              test_id=test_id,
          ),
          answer="",
      )

    end_time = datetime.now(timezone.utc)

    usage_data = getattr(response, "usage_metadata", None)
    logger.debug(f"Response usage: {usage_data}")

    input_tokens = (
        getattr(usage_data, "total_input_tokens", 0) if usage_data else 0
    )
    output_tokens = (
        getattr(usage_data, "total_output_tokens", 0) if usage_data else 0
    )
    total_tokens = getattr(usage_data, "total_tokens", 0) if usage_data else 0
    reasoning_tokens = (
        getattr(usage_data, "total_thought_tokens", 0) if usage_data else 0
    )

    response_text = response.text if hasattr(response, "text") else ""

    input_cost_per_token = self.model_config.pricing.input / 1_000_000
    output_cost_per_token = self.model_config.pricing.output / 1_000_000

    prompt_cost = input_tokens * input_cost_per_token
    completion_cost = output_tokens * output_cost_per_token
    reasoning_cost = reasoning_tokens * output_cost_per_token

    input_choices = [
        Choice(
            index=i, message=Message(role=msg["role"], content=msg["content"])
        )
        for i, msg in enumerate(messages)
    ]
    response_choices = [
        Choice(
            index=len(input_choices),
            message=Message(role="assistant", content=response_text),
        )
    ]
    all_choices = input_choices + response_choices

    metadata = AttemptMetadata(
        model=self.model_config.model_name,
        provider=self.model_config.provider,
        start_timestamp=start_time,
        end_timestamp=end_time,
        choices=all_choices,
        kwargs=self.model_config.kwargs,
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                accepted_prediction_tokens=output_tokens,
                rejected_prediction_tokens=0,
            ),
        ),
        cost=Cost(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            reasoning_cost=reasoning_cost,
            total_cost=prompt_cost + completion_cost + reasoning_cost,
        ),
        task_id=task_id,
        pair_index=pair_index,
        test_id=test_id,
    )
    attempt = Attempt(metadata=metadata, answer=response_text)
    return attempt

  def _build_input_from_messages(self, messages: list) -> str:
    """Convert message list to Interactions API input format.

    For simple prompts, just return the text content.
    """
    for msg in messages:
      if msg.get("role") == "user":
        return msg.get("content", "")
    return ""

  def _build_generation_config(
      self, exclude_keys: Optional[List[str]] = None
  ) -> dict:
    """Build generation_config dict for the Interactions API."""
    exclude_keys = exclude_keys or []
    config = {}

    key_mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_output_tokens": "max_output_tokens",
        "stop_sequences": "stop_sequences",
        "thinking_level": "thinking_level",
        "thinking_summaries": "thinking_summaries",
    }

    for src_key, dest_key in key_mapping.items():
      if src_key in self.generation_config_dict and src_key not in exclude_keys:
        config[dest_key] = self.generation_config_dict[src_key]

    return config

  def chat_completion(self, messages: list):
    """Make an API call using the Interactions API."""
    input_data = self._build_input_from_messages(messages)
    generation_config = self._build_generation_config()

    system_instruction = self.generation_config_dict.get("system_instruction")

    try:
      interaction = self.client.interactions.create(
          model=self.model_config.model_name,
          input=input_data,
          generation_config=generation_config if generation_config else None,
          system_instruction=system_instruction,
          background=self.background_mode,
          store=self.background_mode,
      )

      if self.background_mode:
        interaction = self._poll_for_completion(interaction)

      return _InteractionResponse(interaction)

    except Exception as e:
      logger.error(f"Error in chat_completion with Interactions API: {e}")
      if hasattr(e, "response") and e.response:
        logger.error(f"API Error details: {e.response}")
      return None

  def _poll_for_completion(self, interaction):
    """Poll for background mode completion."""
    while True:
      interaction = self.client.interactions.get(interaction.id)
      logger.debug(f"Interaction status: {interaction.status}")

      if interaction.status == "completed":
        return interaction
      elif interaction.status in ["failed", "error"]:
        raise RuntimeError(
            f"Interaction failed with status: {interaction.status}"
        )

      time.sleep(self.poll_interval)

  def chat_completion_stream(self, messages: list):
    """Make a streaming API call using the Interactions API."""
    logger.debug(
        f"Starting streaming for Gemini model: {self.model_config.model_name}"
    )

    input_data = self._build_input_from_messages(messages)
    generation_config = self._build_generation_config(exclude_keys=["stream"])

    system_instruction = self.generation_config_dict.get("system_instruction")

    try:
      stream = self.client.interactions.create(
          model=self.model_config.model_name,
          input=input_data,
          generation_config=generation_config if generation_config else None,
          system_instruction=system_instruction,
          stream=True,
          store=False,
      )

      text_chunks = []
      usage = None
      chunk_count = 0

      for chunk in stream:
        chunk_count += 1

        if chunk.event_type == "content.delta":
          if hasattr(chunk.delta, "text") and chunk.delta.text:
            text_chunks.append(chunk.delta.text)
        elif chunk.event_type == "interaction.complete":
          usage = getattr(chunk.interaction, "usage", None)

        if chunk_count % 100 == 0:
          logger.debug(f"Streaming progress: {chunk_count} chunks received")

      final_text = "".join(text_chunks)

      logger.debug(
          "Streaming complete for Gemini model:"
          f" {self.model_config.model_name}. Total chunks: {chunk_count}"
      )

      return _StreamResponse(text=final_text, usage_metadata=usage)

    except Exception as e:
      logger.error(
          f"Error in chat_completion_stream with Interactions API: {e}"
      )
      if hasattr(e, "response") and e.response:
        logger.error(f"API Error details: {e.response}")
      return None

  def extract_json_from_response(
      self, input_response: str
  ) -> Optional[List[List[int]]]:
    """Extract JSON from a model response using the Interactions API."""
    prompt = f"""
        Extract only the JSON of the test output from the following response.
        Remove any markdown code blocks and return only valid JSON.

        Response:
        {input_response}

        The JSON should be in this format:
        {{
            "response": [
                [1, 2, 3],
                [4, 5, 6]
            ]
        }}
        """

    extract_config = {}
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
    ]:
      if key in self.generation_config_dict:
        extract_config[key] = self.generation_config_dict[key]

    try:
      interaction = self.client.interactions.create(
          model=self.model_config.model_name,
          input=prompt,
          generation_config=extract_config if extract_config else None,
          store=False,
      )

      response = _InteractionResponse(interaction)
      content = response.text.strip()

      if content.startswith("```json"):
        content = content[7:].strip()
      if content.endswith("```"):
        content = content[:-3].strip()

      try:
        json_data = json.loads(content)
        return json_data.get("response")
      except json.JSONDecodeError:
        logger.error(
            f"Failed to decode JSON from extraction response: {content}"
        )
        return None

    except Exception as e:
      logger.error(
          f"Error in extract_json_from_response with Interactions API: {e}"
      )
      return None
