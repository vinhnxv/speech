"""Standalone test of parsing functions from granite_speech_plus_torch.py.

Tests the parsing logic without needing the model loaded.
"""
import re
import json

# Replicate the parsing functions from the script

def parse_asr(text):
    return {"transcript": text.strip()}


def parse_saa(text):
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
    return {"speakers": speakers, "speaker_count": len(speaker_ids), "raw": text.strip()}


def parse_timestamps(text):
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


_PUNCT = ".,!?;:\"'()[]"


def parse_combined(saa_result, ts_result):
    saa_turns = saa_result["speakers"]
    ts_words = ts_result["words"]
    saa_words_flat = []
    saa_word_to_turn = []
    for turn_idx, turn in enumerate(saa_turns):
        for w in turn["text"].split():
            saa_words_flat.append(w.lower().strip(_PUNCT))
            saa_word_to_turn.append(turn["speaker_id"])
    ts_words_norm = []
    for w in ts_words:
        ts_words_norm.append(w["text"].lower().strip(_PUNCT))
    ts_non_silence = [(i, w) for i, w in enumerate(ts_words) if not w["is_silence"]]
    ts_norm_non_silence = [ts_words_norm[i] for i, _ in ts_non_silence]
    alignment_method = None
    combined_words = []
    if len(saa_words_flat) == len(ts_norm_non_silence):
        alignment_method = "exact_match"
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
            for turn_idx, turn in enumerate(saa_turns):
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


# --- Tests ---

def test_asr():
    result = parse_asr("hello world how are you")
    assert result["transcript"] == "hello world how are you"
    print("✓ ASR parse test passed")


def test_saa():
    text = "[Speaker 1]: hello how are you [Speaker 2]: I am fine [Speaker 1]: great"
    result = parse_saa(text)
    assert result["speaker_count"] == 2
    assert len(result["speakers"]) == 3
    assert result["speakers"][0]["speaker_id"] == 1
    assert result["speakers"][0]["text"] == "hello how are you"
    assert result["speakers"][0]["turn_order"] == 0
    assert result["speakers"][1]["speaker_id"] == 2
    assert result["speakers"][1]["text"] == "I am fine"
    assert result["speakers"][1]["turn_order"] == 1
    assert result["speakers"][2]["speaker_id"] == 1
    assert result["speakers"][2]["text"] == "great"
    print("✓ SAA parse test passed")


def test_saa_single_speaker():
    text = "[Speaker 1]: hello world"
    result = parse_saa(text)
    assert result["speaker_count"] == 1
    assert result["speakers"][0]["speaker_id"] == 1
    print("✓ SAA single speaker test passed")


def test_timestamp_rollover():
    # Simulate audio >10s: word at 0.45s, 0.82s, silence at 9.5s,
    # then word at 0.5s (rolls over to 10.5s), word at 1.0s (rolls over to 11.0s)
    text = "hello [T:45] world [T:82] _ [T:950] test [T:50] word [T:100]"
    result = parse_timestamps(text)
    assert len(result["words"]) == 5
    assert result["words"][0]["text"] == "hello"
    assert result["words"][0]["end_time"] == 0.45
    assert result["words"][1]["text"] == "world"
    assert result["words"][1]["end_time"] == 0.82
    # Silence marker
    assert result["words"][2]["text"] == "_"
    assert result["words"][2]["is_silence"] == True
    assert result["words"][2]["end_time"] == 9.5
    # Rollover: [T:50] = 0.5s + offset 10 = 10.5s
    assert result["words"][3]["text"] == "test"
    assert result["words"][3]["end_time"] == 10.5
    # [T:100] = 1.0s + offset 10 = 11.0s
    assert result["words"][4]["text"] == "word"
    assert result["words"][4]["end_time"] == 11.0
    # Verify at least one word exceeds 10.0
    assert any(w["end_time"] > 10.0 for w in result["words"])
    print("✓ Timestamp rollover test passed")


def test_timestamp_silence():
    text = "hello [T:50] _ [T:80] world [T:120]"
    result = parse_timestamps(text)
    silence_words = [w for w in result["words"] if w["is_silence"]]
    assert len(silence_words) == 1
    assert silence_words[0]["text"] == "_"
    assert silence_words[0]["end_time"] == 0.8
    print("✓ Timestamp silence marker test passed")


def test_timestamp_short_audio():
    # Audio <10s: no rollover needed
    text = "hello [T:50] world [T:100]"
    result = parse_timestamps(text)
    assert result["words"][0]["end_time"] == 0.5
    assert result["words"][1]["end_time"] == 1.0
    assert all(w["end_time"] < 10.0 for w in result["words"])
    print("✓ Timestamp short audio (no rollover) test passed")


def test_combined_exact_match():
    saa_text = "[Speaker 1]: hello world [Speaker 2]: test word"
    saa_result = parse_saa(saa_text)
    ts_text = "hello [T:50] world [T:100] _ [T:150] test [T:200] word [T:250]"
    ts_result = parse_timestamps(ts_text)
    combined = parse_combined(saa_result, ts_result)
    assert combined["alignment_method"] == "exact_match"
    assert combined["saa_word_count"] == 4
    assert combined["ts_word_count"] == 4
    # All 5 words present: hello, world, _, test, word
    assert len(combined["words"]) == 5
    # hello → speaker 1
    assert combined["words"][0]["text"] == "hello"
    assert combined["words"][0]["speaker_id"] == 1
    # world → speaker 1
    assert combined["words"][1]["text"] == "world"
    assert combined["words"][1]["speaker_id"] == 1
    # silence _ inherits speaker 1 (last non-silence speaker)
    assert combined["words"][2]["text"] == "_"
    assert combined["words"][2]["is_silence"] == True
    assert combined["words"][2]["speaker_id"] == 1
    # test → speaker 2
    assert combined["words"][3]["text"] == "test"
    assert combined["words"][3]["speaker_id"] == 2
    # word → speaker 2
    assert combined["words"][4]["text"] == "word"
    assert combined["words"][4]["speaker_id"] == 2
    print("✓ Combined exact match test passed")


def test_combined_proportional_fallback():
    # SAA has more words than timestamps → proportional
    saa_text = "[Speaker 1]: hello world foo [Speaker 2]: test word"
    saa_result = parse_saa(saa_text)
    ts_text = "hello [T:50] world [T:100] test [T:200] word [T:250]"
    ts_result = parse_timestamps(ts_text)
    combined = parse_combined(saa_result, ts_result)
    assert combined["alignment_method"] == "proportional_fallback"
    assert combined["saa_word_count"] == 5
    assert combined["ts_word_count"] == 4
    print("✓ Combined proportional fallback test passed")


def test_combined_unaligned():
    # Edge case: no SAA words
    saa_result = {"speakers": [], "speaker_count": 0, "raw": ""}
    ts_text = "hello [T:50] world [T:100]"
    ts_result = parse_timestamps(ts_text)
    combined = parse_combined(saa_result, ts_result)
    assert combined["alignment_method"] == "unaligned_fallback"
    assert all(w["speaker_id"] is None for w in combined["words"])
    print("✓ Combined unaligned fallback test passed")


def test_json_file_writing():
    import os
    import tempfile

    # Simulate the JSON output structure
    result = {
        "audio": {"path": "test.mp3", "duration": 10.0, "sample_rate": 16000, "channels": 1},
        "session": {"model": "test-model", "mode": "asr", "processing_time": 1.5},
        "transcript": "hello world",
    }
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    json_path = os.path.splitext(tmp_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # Verify file exists and is valid JSON
    assert os.path.isfile(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["transcript"] == "hello world"
    assert loaded["audio"]["duration"] == 10.0
    os.unlink(tmp_path)
    os.unlink(json_path)
    print("✓ JSON file writing test passed")


if __name__ == "__main__":
    test_asr()
    test_saa()
    test_saa_single_speaker()
    test_timestamp_rollover()
    test_timestamp_silence()
    test_timestamp_short_audio()
    test_combined_exact_match()
    test_combined_proportional_fallback()
    test_combined_unaligned()
    test_json_file_writing()
    print("\n✅ All tests passed!")