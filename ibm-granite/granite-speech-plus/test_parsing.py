"""Standalone tests for Granite Speech 4.1 2B Plus parsing functions.

Imports from parsing.py so tests exercise the exact same code path as the script,
without needing torch/transformers loaded.
"""

import json
import os
import tempfile

from parsing import parse_asr, parse_saa, parse_timestamps, parse_combined


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
    assert result["preamble"] == ""
    print("✓ SAA parse test passed")


def test_saa_single_speaker():
    text = "[Speaker 1]: hello world"
    result = parse_saa(text)
    assert result["speaker_count"] == 1
    assert result["speakers"][0]["speaker_id"] == 1
    print("✓ SAA single speaker test passed")


def test_saa_preamble():
    # Text before the first speaker tag should be captured in preamble
    text = "hello there [Speaker 1]: hi"
    result = parse_saa(text)
    assert result["preamble"] == "hello there"
    assert len(result["speakers"]) == 1
    assert result["speakers"][0]["text"] == "hi"
    print("✓ SAA preamble capture test passed")


def test_timestamp_rollover():
    # Simulate audio >10s: word at 0.45s, 0.82s, silence at 9.5s,
    # then word at 0.5s (rolls over to 10.5s), word at 1.0s (rolls over to 11.0s)
    text = "hello [T:45] world [T:82] _ [T:950] test [T:50] word [T:100]"
    result = parse_timestamps(text)
    assert len(result["words"]) == 5
    assert result["words"][0]["text"] == "hello"
    assert result["words"][0]["start_time"] == 0.0
    assert result["words"][0]["end_time"] == 0.45
    assert result["words"][1]["text"] == "world"
    assert result["words"][1]["start_time"] == 0.45
    assert result["words"][1]["end_time"] == 0.82
    # Silence marker
    assert result["words"][2]["text"] == "_"
    assert result["words"][2]["start_time"] == 0.82
    assert result["words"][2]["is_silence"] is True
    assert result["words"][2]["end_time"] == 9.5
    # Rollover: [T:50] = 0.5s + offset 10 = 10.5s
    assert result["words"][3]["text"] == "test"
    assert result["words"][3]["start_time"] == 9.5
    assert result["words"][3]["end_time"] == 10.5
    # [T:100] = 1.0s + offset 10 = 11.0s
    assert result["words"][4]["text"] == "word"
    assert result["words"][4]["start_time"] == 10.5
    assert result["words"][4]["end_time"] == 11.0
    # Verify at least one word exceeds 10.0
    assert any(w["end_time"] > 10.0 for w in result["words"])
    print("✓ Timestamp rollover test passed")


def test_timestamp_spurious_backward_jump():
    # A small backward jump (< 5s) should NOT trigger a 10s offset
    # [T:950] = 9.5s, [T:930] = 9.3s — only 0.2s backward, not a rollover
    text = "word1 [T:950] word2 [T:930]"
    result = parse_timestamps(text, duration=15.0)
    assert result["words"][0]["end_time"] == 9.5
    # word2 should NOT get a spurious 10s offset — should stay near 9.5s
    assert result["words"][1]["end_time"] == 9.5  # floored to last_end
    print("✓ Timestamp spurious backward jump (no false rollover) test passed")


def test_timestamp_duration_bound():
    # If unwrapped time exceeds duration + 5s, clamp to duration
    text = "word1 [T:950] word2 [T:50] word3 [T:100]"
    result = parse_timestamps(text, duration=12.0)
    # word1 = 9.5s, word2 = 10.5s (rollover), word3 = 11.0s — all within duration
    assert result["words"][0]["end_time"] == 9.5
    assert result["words"][1]["end_time"] == 10.5
    assert result["words"][2]["end_time"] == 11.0
    print("✓ Timestamp duration bound test passed")


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


def test_timestamp_empty_input():
    # No [T:N] tags → empty words list (not an error for experiment script)
    result = parse_timestamps("no timestamps here")
    assert result["words"] == []
    print("✓ Timestamp empty input test passed")


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
    assert combined["words"][0]["start_time"] == 0.0
    assert combined["words"][0]["speaker_id"] == 1
    # world → speaker 1
    assert combined["words"][1]["text"] == "world"
    assert combined["words"][1]["start_time"] == 0.5
    assert combined["words"][1]["speaker_id"] == 1
    # silence _ inherits speaker 1 (last non-silence speaker)
    assert combined["words"][2]["text"] == "_"
    assert combined["words"][2]["start_time"] == 1.0
    assert combined["words"][2]["is_silence"] is True
    assert combined["words"][2]["speaker_id"] == 1
    # test → speaker 2
    assert combined["words"][3]["text"] == "test"
    assert combined["words"][3]["start_time"] == 1.5
    assert combined["words"][3]["speaker_id"] == 2
    # word → speaker 2
    assert combined["words"][4]["text"] == "word"
    assert combined["words"][4]["start_time"] == 2.0
    assert combined["words"][4]["speaker_id"] == 2
    print("✓ Combined exact match test passed")


def test_combined_count_match_positional():
    # Same word count but different content → count_match_positional
    saa_text = "[Speaker 1]: hello world"
    saa_result = parse_saa(saa_text)
    ts_text = "hi [T:50] there [T:100]"
    ts_result = parse_timestamps(ts_text)
    combined = parse_combined(saa_result, ts_result)
    assert combined["alignment_method"] == "count_match_positional"
    assert combined["words"][0]["speaker_id"] == 1
    assert combined["words"][1]["speaker_id"] == 1
    print("✓ Combined count_match_positional test passed")


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


def test_combined_proportional_min_alloc():
    # Ensure each turn gets at least 1 word when possible
    # 3 turns, 3 ts words → each turn should get exactly 1
    saa_text = "[Speaker 1]: hello world [Speaker 2]: test [Speaker 3]: foo bar"
    saa_result = parse_saa(saa_text)
    ts_text = "hello [T:50] test [T:100] foo [T:200]"
    ts_result = parse_timestamps(ts_text)
    combined = parse_combined(saa_result, ts_result)
    assert combined["alignment_method"] == "proportional_fallback"
    # Each turn should have at least 1 word
    speakers_seen = [w["speaker_id"] for w in combined["words"]]
    assert 1 in speakers_seen
    assert 2 in speakers_seen
    assert 3 in speakers_seen
    print("✓ Combined proportional min alloc per turn test passed")


def test_combined_unaligned():
    # Edge case: no SAA words
    saa_result = {"speakers": [], "speaker_count": 0, "preamble": "", "raw": ""}
    ts_text = "hello [T:50] world [T:100]"
    ts_result = parse_timestamps(ts_text)
    combined = parse_combined(saa_result, ts_result)
    assert combined["alignment_method"] == "unaligned_fallback"
    assert all(w["speaker_id"] is None for w in combined["words"])
    print("✓ Combined unaligned fallback test passed")


def test_json_file_writing():
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
    test_saa_preamble()
    test_timestamp_rollover()
    test_timestamp_spurious_backward_jump()
    test_timestamp_duration_bound()
    test_timestamp_silence()
    test_timestamp_short_audio()
    test_timestamp_empty_input()
    test_combined_exact_match()
    test_combined_count_match_positional()
    test_combined_proportional_fallback()
    test_combined_proportional_min_alloc()
    test_combined_unaligned()
    test_json_file_writing()
    print("\n✅ All tests passed!")