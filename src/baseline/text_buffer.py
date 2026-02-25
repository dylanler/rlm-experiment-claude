"""
Text Buffer Baseline: RLM-style text-buffer approach for comparison.
Each chunk is summarized to text, then all summaries are concatenated
and fed with the question for final answer generation.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class TextBufferBaseline:
    """
    For each chunk:
      1. Feed chunk + task prompt to LM
      2. Generate a text summary/extraction
      3. Store text in buffer
    After all chunks:
      4. Concatenate all text buffers (truncate if needed)
      5. Feed concatenated buffer + question to LM
      6. Generate final answer
    """

    def __init__(self, model, tokenizer, chunk_size=1024, max_buffer_tokens=4096):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_buffer_tokens = max_buffer_tokens

    def process_chunk(self, chunk_text: str, task_prompt: str) -> str:
        """Generate a text summary/extraction for a single chunk."""
        prompt = (
            f"{task_prompt}\n\n"
            f"Document section:\n{chunk_text}\n\n"
            f"Extracted information:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.chunk_size + 512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def aggregate_and_answer(self, buffers: list[str], question: str) -> str:
        """Concatenate text buffers and generate final answer."""
        combined = "\n---\n".join(buffers)
        # Truncate to max_buffer_tokens if needed
        combined_ids = self.tokenizer(
            combined, truncation=True, max_length=self.max_buffer_tokens
        )
        combined_text = self.tokenizer.decode(
            combined_ids.input_ids, skip_special_tokens=True
        )

        prompt = (
            f"Based on the following extracted information:\n{combined_text}\n\n"
            f"Question: {question}\nAnswer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_buffer_tokens + 512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def run(
        self,
        document: str,
        question: str,
        chunks: list[dict],
        task_prompt: str = "Extract all key information from the following document section that could be relevant to answering questions about the document.",
    ) -> str:
        """Full pipeline: chunk -> summarize each -> aggregate -> answer."""
        buffers = []
        for chunk in chunks:
            logger.debug(f"Processing chunk {chunk['chunk_id']}")
            summary = self.process_chunk(chunk["text"], task_prompt)
            buffers.append(summary)

        answer = self.aggregate_and_answer(buffers, question)
        return answer
