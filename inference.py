import torch
from torch.nn import functional as F
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from contextlib import contextmanager
import threading
import itertools
import gc

# <|endoftext|> tokens
pad_token_id = 50256
eos_token_id = 50256

class ModelInfo():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

class ModelCacheEntry():
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.usage_count = 1

class ModelManager():
    lock = threading.Lock()
    cache = {}

    @contextmanager
    def use_model(self, model_name: str):
        with ModelManager.lock:
            if model_name in ModelManager.cache:
                ModelManager.cache[model_name].usage_count += 1
                model_info = ModelManager.cache[model_name].model_info
            else:
                self.free_unused_models()
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                model = GPT2LMHeadModel.from_pretrained(model_name)
                model_info = ModelInfo(model, tokenizer)
                ModelManager.cache[model_name] = ModelCacheEntry(model_info)
        try:
            yield model_info
        except Exception as e:
            raise e
        finally:
            with ModelManager.lock:
                if model_name in ModelManager.cache:
                    ModelManager.cache[model_name].usage_count -= 1

    def free_unused_models(self):
        for model_name in list(ModelManager.cache.keys()):
            if ModelManager.cache[model_name].usage_count <= 0:
                    ModelManager.cache[model_name].model = None
                    ModelManager.cache[model_name].tokenizer = None
                    del ModelManager.cache[model_name]
        gc.collect()

class TextSampler():
    def __init__(self, model_info: ModelInfo):
        self.min_length = 10
        self.temperature = 1.0
        self.top_k = 0
        self.top_p = 1.0
        self.model_info = model_info

    # Sample at most num tokens (multi-byte tokens counted as one)
    def sample_text_atmost(self, prefix: str, num):
        return itertools.islice(self.sample_text(prefix), num)

    # A quick wrapper to return a string instead of an iterator
    def generate_text_atmost(self, prefix: str, num):
        return ''.join(self.sample_text_atmost(prefix, num))

    @torch.no_grad()
    def sample_text(self, prefix: str):
        model = self.model_info.model
        tokenizer = self.model_info.tokenizer
        input_ids = tokenizer(prefix, return_tensors='pt')['input_ids']

        # Context_ids keeps the last 512 tokens generated during sampling
        # This is used for generating sequences longer than the gpt2 context limit (1024 tokens)
        context_ids = input_ids[:,-512:]

        # Set up logits processors for basic sampling task
        logits_processors = LogitsProcessorList()

        # Prevent EOS from being generated before min_length is reached
        logits_processors.append(MinLengthLogitsProcessor(self.min_length, eos_token_id))
        # Use Temperature modifier to modify distribution
        if self.temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(self.temperature))
        # Use top_k filtering
        if self.top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k=self.top_k, min_tokens_to_keep=1))
        # Use top_p filtering
        if self.top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=1))

        # Past tracks previous weighting to be reused
        past=None
        # Output decoding buffer
        decode_buff = torch.tensor([])

        while(True):
            outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True, return_dict=True, output_attentions=False, output_hidden_states=False)
            next_token_logits = outputs.logits[:, -1, :]
            # Gets destribution after processing with different filters
            scores = logits_processors(input_ids, next_token_logits)
            # Normalize with softmax
            prob_destribution = F.softmax(scores, dim=-1)
            # Sample an output
            next_token = torch.multinomial(prob_destribution, num_samples=1).squeeze(1)
            # Check if next token is eos
            if next_token == eos_token_id:
                # Stops execution
                return
            # The next input token is the output
            # Since we used the past value, we only need to past in the newest token
            input_ids = next_token[:, None]
            # Save next token to context_ids
            context_ids = torch.cat([context_ids[:,-511:],next_token[:, None]], dim=-1)
            # Tries to decode the tokens got so far
            decode_buff = torch.cat([decode_buff,next_token], dim=-1)
            decoded = tokenizer.decode(decode_buff)
            # Some characters span multiple tokens (BPE)
            # Therefore, only yield a result when decode is successful
            if not decoded.endswith('\uFFFD') or len(decode_buff) > 20:
                yield decoded
                decode_buff = torch.tensor([])
            # Update past to save computation time
            past = outputs.past_key_values
            # The maximum context of gpt2 is 1024 tokens
            if past[0].size()[3] >= 1024:
                # If the sequence length has exceeded, discard past and use context_ids
                past = None
                input_ids = context_ids