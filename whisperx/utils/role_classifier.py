"""
utils/role_classifier.py
Optional LLM-based speaker role classifier.

Uses GPT-4o (or any OpenAI-compatible API) to classify each
speaker as doctor / patient / attendant based on their first
~120 words of transcript.

Usage:
    from utils.role_classifier import llm_role_map
    role_map = llm_role_map(utterances, api_key="sk-...")
"""

import json
from collections import defaultdict
from typing import List, Dict


def _build_speaker_sample(utterances: List[dict], max_words: int = 120) -> Dict[str, str]:
    samples = defaultdict(list)
    counts  = defaultdict(int)
    for u in utterances:
        spk = u["speaker"]
        if counts[spk] < max_words:
            words = u["text"].split()
            take  = words[: max_words - counts[spk]]
            samples[spk].extend(take)
            counts[spk] += len(take)
    return {spk: " ".join(wds) for spk, wds in samples.items()}


def llm_role_map(
    utterances: List[dict],
    api_key: str,
    model: str = "gpt-4o",
) -> Dict[str, str]:
    """
    Classify each speaker using an LLM.
    Returns a dict like {"SPEAKER_00": "doctor", "SPEAKER_01": "patient", ...}

    Falls back to an empty dict on any error — caller should
    then fall back to heuristic_role_map().
    """
    try:
        import openai
    except ImportError:
        print("openai package not found. Run: pip install openai")
        return {}

    client  = openai.OpenAI(api_key=api_key)
    samples = _build_speaker_sample(utterances)

    prompt_lines = [
        f"Speaker {spk}:\n{text}\n"
        for spk, text in samples.items()
    ]

    system_msg = (
        "You are a clinical conversation analyst. "
        "Given transcript excerpts from a doctor–patient consultation in Bengali, "
        "classify each speaker as exactly one of: doctor, patient, attendant. "
        "Reply ONLY with a JSON object like: "
        '{"SPEAKER_00": "doctor", "SPEAKER_01": "patient", "SPEAKER_02": "attendant"}'
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": "\n".join(prompt_lines)},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        result = json.loads(resp.choices[0].message.content)
        print(f"LLM role classification: {result}")
        return result
    except Exception as e:
        print(f"LLM classification failed: {e}")
        return {}
