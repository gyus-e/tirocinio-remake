from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(self, model_name: str):
        self._device = Accelerator().device
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model(self) -> AutoModelForCausalLM:
        return self._model

    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def device(self):
        return self._device
