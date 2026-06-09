"""Generation and critic prompt templates (spec section 5.3.2).

Each sample type has a generation template that must emit JSON only, and a
critic template that scores the result. Templates use ``{chunk_text}`` and,
for critics, the generated fields.
"""
from __future__ import annotations

QA_GENERATION = """You are a training data author. Given the passage below, write one \
high-quality question and a thorough answer. The question must be answerable solely \
from the passage. Do not copy the passage verbatim. Output JSON only:
{{"question": "...", "answer": "..."}}

PASSAGE:
{chunk_text}"""

QA_CRITIC = """You are a dataset quality reviewer. Evaluate the following Q&A pair \
against the source passage. Score each dimension 1-5 and return JSON only:
{{"faithfulness": N, "completeness": N, "clarity": N, "reject": false, "reason": "..."}}

SOURCE: {chunk_text}
Q: {question}
A: {answer}"""

INSTRUCTION_GENERATION = """You are a training data author. Read the passage below and \
write one realistic task instruction that the passage could serve as a high-quality \
response to, plus that response. The response must be grounded in the passage. \
Output JSON only:
{{"instruction": "...", "input": "", "output": "..."}}

PASSAGE:
{chunk_text}"""

INSTRUCTION_CRITIC = """You are a dataset quality reviewer. Evaluate whether the output \
is a faithful, complete, clear response to the instruction and is grounded in the \
source passage. Score each dimension 1-5 and return JSON only:
{{"faithfulness": N, "completeness": N, "clarity": N, "reject": false, "reason": "..."}}

SOURCE: {chunk_text}
INSTRUCTION: {instruction}
OUTPUT: {output}"""


DEFAULT_TEMPLATES: dict[str, str] = {
    "qa_generation": QA_GENERATION,
    "qa_critic": QA_CRITIC,
    "instruction_generation": INSTRUCTION_GENERATION,
    "instruction_critic": INSTRUCTION_CRITIC,
}

# Human-readable labels for the UI.
SAMPLE_TYPE_LABELS = {
    "qa": "Q&A",
    "instruction": "Instruction-following",
}
