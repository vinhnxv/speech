"""Parsing functions for Granite Speech 4.1 2B Plus output.

Shared by granite_speech_plus_torch.py and test_parsing.py so that tests
exercise the exact same code path as the script without importing torch/transformers.
"""

import re

_PUNCT = ".,!?;:\"'()[]"


def parse_asr(text):
    """Parse plain ASR output — just a transcript string."""
    return {"transcript": text.strip()}


def parse_saa(text):
    """Parse speaker-attributed ASR output into speaker turns.

    Output format: [Speaker 1]: text [Speaker 2]: text ...
    """
    parts = re.split(r"(\[Speaker \d+\]:)", text)
    speakers = []
    speaker_ids = set()
    current_speaker = None
    current_text = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"\[Speaker (\d+)\]:", part)
        if match:
            if current_speaker is not None:
                speakers.append({
                    "speaker_id": current_speaker,
                    "text": current_text.strip(),
                    "turn_order": len(speakers),
                })
            current_speaker = int(match.group(1))
            speaker_ids.add(current_speaker)
            current_text = ""
        else:
            current_text = (current_text + " " + part).strip() if current_text else part

    if current_speaker is not None:
        speakers.append({
            "speaker_id": current_speaker,
            "text": current_text.strip(),
            "turn_order": len(speakers),
        })

    return {
        "speakers": speakers,
        "speaker_count": len(speaker_ids),
        "raw": text.strip(),
    }


def parse_timestamps(text):
    """Parse word-level timestamp output into per-word records.

    Timestamps are in centiseconds mod 1000 (10s rollover).
    N = round(t * 100) mod 1000; recover with t = N/100 + 10*R.
    Silence is transcribed as `_`.
    """
    ts_parts = re.split(r"\[T:(\d+)\]", text)
    words = []
    last_end = 0.0
    offset = 0.0

    word_parts = ts_parts[::2]
    tag_parts = ts_parts[1::2]

    for word_text, ts in zip(word_parts, tag_parts):
        word_text = word_text.strip()
        if not word_text:
            continue

        raw_time = float(ts) / 100.0
        while raw_time + offset < last_end:
            offset += 10.0
        abs_time = raw_time + offset
        last_end = abs_time

        is_silence = word_text == "_"

        words.append({
            "text": word_text,
            "end_time": round(abs_time, 2),
            "is_silence": is_silence,
        })

    return {"words": words}


def parse_combined(saa_result, ts_result):
    """Align speaker turns with word timings by text matching.

    Runs SAA and timestamps separately, then aligns by normalizing word sequences.
    """
    saa_turns = saa_result["speakers"]
    ts_words = ts_result["words"]

    # Flatten SAA turns into a normalized word list (lowercase, no tags)
    saa_words_flat = []
    saa_word_to_turn = []
    for turn in saa_turns:
        for w in turn["text"].split():
            saa_words_flat.append(w.lower().strip(_PUNCT))
            saa_word_to_turn.append(turn["speaker_id"])

    # Normalize timestamp words
    ts_words_norm = [w["text"].lower().strip(_PUNCT) for w in ts_words]

    # Non-silence words for alignment (silence markers don't correspond to SAA words)
    ts_non_silence = [(i, w) for i, w in enumerate(ts_words) if not w["is_silence"]]
    ts_norm_non_silence = [ts_words_norm[i] for i, _ in ts_non_silence]

    alignment_method = None
    combined_words = []

    if len(saa_words_flat) == len(ts_norm_non_silence):
        # Word count matches — check if content also matches
        content_matches = all(a == b for a, b in zip(saa_words_flat, ts_norm_non_silence))
        alignment_method = "exact_match" if content_matches else "count_match_positional"

        saa_idx = 0
        last_speaker = None
        for ts_w in ts_words:
            if ts_w["is_silence"]:
                speaker_id = last_speaker
            else:
                speaker_id = saa_word_to_turn[saa_idx]
                saa_idx += 1
                last_speaker = speaker_id
            combined_words.append({
                "text": ts_w["text"],
                "end_time": ts_w["end_time"],
                "is_silence": ts_w["is_silence"],
                "speaker_id": speaker_id,
            })
    else:
        # Proportional allocation: distribute ts words across saa turns by turn length ratio
        alignment_method = "proportional_fallback"
        total_saa_words = len(saa_words_flat)
        total_ts_words = len(ts_norm_non_silence)

        if total_saa_words == 0 or total_ts_words == 0:
            alignment_method = "unaligned_fallback"
            for ts_w in ts_words:
                combined_words.append({
                    "text": ts_w["text"],
                    "end_time": ts_w["end_time"],
                    "is_silence": ts_w["is_silence"],
                    "speaker_id": None,
                })
        else:
            ts_idx = 0
            for turn in saa_turns:
                turn_word_count = len(turn["text"].split())
                alloc = round(turn_word_count * total_ts_words / total_saa_words)
                alloc = max(1, alloc)

                allocated = 0
                while ts_idx < len(ts_words) and allocated < alloc:
                    ts_w = ts_words[ts_idx]
                    combined_words.append({
                        "text": ts_w["text"],
                        "end_time": ts_w["end_time"],
                        "is_silence": ts_w["is_silence"],
                        "speaker_id": turn["speaker_id"],
                    })
                    ts_idx += 1
                    if not ts_w["is_silence"]:
                        allocated += 1

            # Assign remaining ts words to last speaker
            while ts_idx < len(ts_words):
                ts_w = ts_words[ts_idx]
                combined_words.append({
                    "text": ts_w["text"],
                    "end_time": ts_w["end_time"],
                    "is_silence": ts_w["is_silence"],
                    "speaker_id": saa_turns[-1]["speaker_id"] if saa_turns else None,
                })
                ts_idx += 1

    return {
        "words": combined_words,
        "alignment_method": alignment_method,
        "saa_word_count": len(saa_words_flat),
        "ts_word_count": len(ts_norm_non_silence),
    }