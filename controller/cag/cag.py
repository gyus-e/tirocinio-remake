import os
from typing import Optional
import torch
from accelerate import Accelerator
from transformers.cache_utils import DynamicCache
from environ import CACHE_NAME, STORAGE


_accelerator = Accelerator()


def create_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    """prepares a reusable key-value cache for a transformer model's attention mechanism."""
    """passes a prompt through the model once, creating a KV cache that records all the hidden states from each layer"""
    """@param model: the transformer model used for encoding the prompt."""
    """@param tokenizer: the tokenizer to convert the prompt into the token IDs."""
    """@param prompt: a string input used as the prompt"""
    """@return: DynamicCache object containing the key-value cache."""

    torch_device = _accelerator.device
    print(f"Using device: {torch_device}")

    # Tokenize the prompt using the tokenizer and convert it into input IDs
    input_ids: torch.Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(
        torch_device
    )
    print("Prompt tokenized.")

    # Initialize the DynamicCache object
    cache: DynamicCache = DynamicCache()
    print("DynamicCache initialized.")

    # Perform forward pass through the model with caching enabled,
    # populating the cache with key-value pairs resulting from the model's computation
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )

    print("KV cache created.")
    return cache


def save_cache(my_cache: DynamicCache, storage=STORAGE, cache_name=CACHE_NAME) -> None:
    os.makedirs(storage, exist_ok=True)
    cache_path = os.path.join(storage, cache_name)
    torch.save(my_cache, cache_path)


def clean_up_cache(cache: DynamicCache, origin_len: Optional[int] = None) -> None:
    """Cleans the key-value cache by removing unnecessary entries"""
    """Trims a DynamicCache object to match the original sequence length by removing additional tokens added during processing"""
    """For each layer of the cache, it slices both the key and value tensors to retain only the first origin_len tokens along the sequence dimension"""

    if origin_len == None:
        origin_len = _default_cache_len(cache)

    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, origin_len:, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, origin_len:, :]


def _default_cache_len(cache: DynamicCache) -> int:
    return cache.key_cache[0].shape[-2]


def get_answer(
    question: str, tokenizer, model, device: torch.device, loaded_cache: DynamicCache
) -> str:
    # Call generate to produce the answer
    input_ids_q = tokenizer(question + "\n", return_tensors="pt").input_ids.to(device)
    gen_ids_q = _generate(model, input_ids_q, loaded_cache, max_new_tokens=100)

    # Decode the final result with tokenizer.decode
    answer = tokenizer.decode(gen_ids_q[0], skip_special_tokens=True)
    return answer.strip()


def _generate(
    model, input_ids, past_key_values: DynamicCache, max_new_tokens: int = 100
) -> torch.Tensor:
    """The generate function handles token-by-token generation with the cached knowledge using greedy decoding."""
    """Greedy decoding is a simple text generation method where, at each step, the token with the highest probability (maximum value in the logits) is selected as the next token."""
    """Greedy decoding is equivalent to temperature=0 in the context of text generation."""
    """@param model: The LLM."""
    """@param input_ids: A tensor containing the tokenized input sequence."""
    """@param past_key_values: the core component of CAG: a cache of previously computed attention values used to speed up inference by avoiding recomputation."""
    """@param max_new_tokens: The maximum number of new tokens to generate."""
    """@return: A tensor containing the generated token IDs."""

    torch_device = _accelerator.device
    origin_len: int = input_ids.shape[-1]

    input_ids = input_ids.to(torch_device)

    output_ids: torch.Tensor = input_ids.clone()
    next_token: torch.Tensor = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Process current input token in next_token and cached past_key_values
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]

            # Identify the token with the highest probability using greedy decoding
            token = torch.argmax(logits, dim=-1, keepdim=True)

            # This new token is appended to the output sequence
            output_ids = torch.cat([output_ids, token], dim=-1)

            # The cache is updated to include the current context
            past_key_values = out.past_key_values

            # The newly generated token becomes the input for the next iteration
            next_token = token.to(torch_device)

            # Terminate early if an end-of-sequence token is generated
            if (
                model.config.eos_token_id is not None
                and token.item() == model.config.eos_token_id
            ):
                break

    return output_ids[:, origin_len:]
