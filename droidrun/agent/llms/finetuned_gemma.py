"""
Fine-tuned Gemma3n model provider for Android UI tasks.
"""
# Import unsloth first to ensure all optimizations are applied
import logging
import os
from typing import Any, Optional, Sequence

import torch
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.custom import CustomLLM
from unsloth import FastVisionModel

logger = logging.getLogger("droidrun")


class FineTunedGemmaLLM(CustomLLM):
    """Fine-tuned Gemma3n model using LoRA adapters."""

    model_path: str = "finetuning/models/gemma3n-ft-run-v2"
    base_model: str = "google/gemma-3n-e4b-it"
    max_seq_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(**kwargs)

        if model_path:
            self.model_path = model_path
        if base_model:
            self.base_model = base_model

        self.temperature = temperature
        self.top_p = top_p
        self.device = device

        # Load the model with LoRA adapters
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned model with LoRA adapters."""
        logger.info(f"Loading fine-tuned model from {self.model_path}")

        # Check if we're loading from a local path or HuggingFace
        if os.path.exists(self.model_path):
            # Local LoRA adapters
            logger.info("Loading local LoRA adapters...")
            self.model, base_tokenizer = FastVisionModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=torch.float16,
                load_in_4bit=True,
            )

            # Try to load tokenizer from fine-tuned model directory
            tokenizer_path = os.path.join(self.model_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_path):
                logger.info(f"Loading tokenizer from fine-tuned model at {self.model_path}")
                from transformers import AutoProcessor, AutoTokenizer
                try:
                    # Try to load as processor first (for multimodal models)
                    self.tokenizer = AutoProcessor.from_pretrained(self.model_path)
                    logger.info(f"Loaded processor of type {type(self.tokenizer)}")
                except Exception as e:
                    logger.warning(f"Could not load as processor, trying tokenizer: {e}")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                        logger.info(f"Loaded tokenizer of type {type(self.tokenizer)}")
                    except Exception as e2:
                        logger.warning(f"Could not load fine-tuned tokenizer, using base: {e2}")
                        self.tokenizer = base_tokenizer
            else:
                logger.info("No tokenizer found in fine-tuned model directory, using base tokenizer")
                self.tokenizer = base_tokenizer

            # Load the LoRA adapters directly (inference-only)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        else:
            # Load from HuggingFace Hub
            logger.info(f"Loading model from HuggingFace: {self.model_path}")
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.model_path,  # This should be your HF repo
                max_seq_length=self.max_seq_length,
                dtype=torch.float16,
                load_in_4bit=True,
            )

        # Set to inference mode (disable gradients + training mode)
        FastVisionModel.for_inference(self.model)
        self.model.eval()  # Ensure eval mode

        # Verify no gradients are required
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"Model and tokenizer loaded. Tokenizer type: {type(self.tokenizer)}")
        logger.info("✅ Fine-tuned model loaded for inference (no gradients)!")

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata for telemetry and monitoring."""
        return LLMMetadata(
            context_window=self.max_seq_length,
            num_output=512,  # max_new_tokens we use in generation
            model_name=f"FineTunedGemma-{os.path.basename(self.model_path)}",
            is_chat_model=True,
        )

    @property
    def _metadata_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "base_model": self.base_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt using the fine-tuned model."""
        logger.debug(f"Complete called with prompt of length {len(prompt) if prompt else 0}")

        # Handle the tokenizer/processor correctly
        if hasattr(self.tokenizer, 'tokenizer'):
            # It's a processor, use the underlying tokenizer
            logger.debug("Using processor's tokenizer")
            inputs = self.tokenizer.tokenizer(prompt, return_tensors="pt").to(self.device)
        else:
            # It's a regular tokenizer
            logger.debug("Using tokenizer directly")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

        # Debug: Check what was generated
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Full generated response: '{full_response}'")
        logger.debug(f"Original prompt length: {len(prompt)}")

        # Smart response extraction - look for model response after the prompt
        response = full_response

        # For chat-formatted responses, look for the last "model" or assistant marker
        if "model\n" in full_response:
            # Split on model marker and take the last part
            response_parts = full_response.split("model\n")
            if len(response_parts) > 1:
                response = response_parts[-1].strip()
        elif prompt in full_response:
            # Fallback to simple prompt removal if we can find exact prompt
            response = full_response[len(prompt):].strip()
        else:
            # If no clear markers, try to extract meaningful content after template
            response = full_response.strip()

        logger.debug(f"Extracted response: '{response[:200]}...' (length: {len(response)})")

        if not response:
            logger.warning("Model generated empty response! This indicates a problem with the model or prompt format.")
            logger.warning(f"Prompt was: {prompt[:200]}...")
            # Return the full response as fallback
            response = full_response

        return CompletionResponse(text=response)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion - for now just yield the full response."""
        response = self._complete(prompt, **kwargs)
        yield response

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Public completion method."""
        return self._complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """Public stream completion method."""
        return self._stream_complete(prompt, **kwargs)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat using the fine-tuned model with minimal context."""
        logger.debug(f"Chat called with {len(messages)} messages")

        # For fine-tuned models, we want MINIMAL context:
        # 1. System message (task specification)
        # 2. Only the most recent user message (with screenshot)

        system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]

        # Take system message and ONLY the last user message
        essential_messages = []

        if system_messages:
            essential_messages.extend(system_messages)  # Keep all system messages

        if user_messages:
            # Only keep the most recent user message (which should contain current screenshot + task summary)
            essential_messages.append(user_messages[-1])

        logger.info(f"FineTuned model: Reduced {len(messages)} messages to {len(essential_messages)} essential messages")

        # Convert messages to the model's expected format
        formatted_messages = []
        for msg in essential_messages:
            content = msg.content if msg.content is not None else ""
            logger.debug(f"Essential message: role={msg.role}, content_length={len(content)}")

            if msg.role == MessageRole.SYSTEM:
                formatted_messages.append({"role": "system", "content": content})
            elif msg.role == MessageRole.USER:
                formatted_messages.append({"role": "user", "content": content})
            elif msg.role == MessageRole.ASSISTANT:
                formatted_messages.append({"role": "assistant", "content": content})

        # Apply chat template
        logger.debug(f"Applying chat template to {len(formatted_messages)} formatted messages")
        logger.debug(f"Tokenizer has apply_chat_template: {hasattr(self.tokenizer, 'apply_chat_template')}")

        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"Chat template returned: {type(prompt)}, length={len(prompt) if prompt else 0}")
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            prompt = None

        # Check if prompt is None or empty
        if not prompt:
            logger.warning("Chat template returned None/empty, constructing manual prompt")
            # Fallback to manual prompt construction
            prompt = ""
            for msg in formatted_messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "

            if not prompt.strip() or prompt == "Assistant: ":
                logger.error("No valid content in messages!")
                prompt = "Hello! How can I help you today?"

        # CLEAN LOGGING FOR FINE-TUNED MODEL - Show what we're asking
        print("\n" + "="*60)
        print("🎯 FINE-TUNED MODEL CONVERSATION")
        print("="*60)

        for msg in essential_messages:
            role_icon = "🤖" if msg.role == MessageRole.SYSTEM else "👤"
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            print(f"{role_icon} {msg.role.name}: {content}")

        print("-" * 60)
        print("⏳ Generating fine-tuned response...")

        # Generate response
        completion = self._complete(prompt)

        # LOG THE RESPONSE
        displayed_response = completion.text[:500] + "..." if len(completion.text) > 500 else completion.text
        print(f"🎯 FINE-TUNED RESPONSE: {displayed_response}")
        print("="*60 + "\n")

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=completion.text
            )
        )
