from podcastpipe.textnorm import (
    chunk_text,
    normalize_for_tts,
    number_to_words,
    ordinal_to_words,
    split_sentences,
    year_to_words,
)


class TestNumbers:
    def test_small_numbers(self):
        assert number_to_words(0) == "zero"
        assert number_to_words(7) == "seven"
        assert number_to_words(19) == "nineteen"

    def test_tens_and_hundreds(self):
        assert number_to_words(20) == "twenty"
        assert number_to_words(42) == "forty-two"
        assert number_to_words(100) == "one hundred"
        assert number_to_words(365) == "three hundred sixty-five"

    def test_thousands(self):
        assert number_to_words(1000) == "one thousand"
        assert number_to_words(1200) == "one thousand two hundred"
        assert number_to_words(15_400) == "fifteen thousand four hundred"

    def test_very_large_falls_back_to_digits(self):
        # A nine-digit number read aloud is a script problem, not ours to fix.
        assert number_to_words(1_234_567) == "1234567"

    def test_ordinals(self):
        assert ordinal_to_words(1) == "first"
        assert ordinal_to_words(2) == "second"
        assert ordinal_to_words(3) == "third"
        assert ordinal_to_words(4) == "fourth"
        assert ordinal_to_words(12) == "twelfth"
        assert ordinal_to_words(20) == "twentieth"
        assert ordinal_to_words(21) == "twenty-first"


class TestYears:
    def test_broadcast_convention(self):
        assert year_to_words(1998) == "nineteen ninety-eight"
        assert year_to_words(2026) == "twenty twenty-six"
        assert year_to_words(1776) == "seventeen seventy-six"

    def test_two_thousands_read_as_two_thousand_n(self):
        assert year_to_words(2005) == "two thousand five"
        assert year_to_words(2000) == "two thousand"

    def test_century_boundaries(self):
        assert year_to_words(1900) == "nineteen hundred"
        assert year_to_words(1907) == "nineteen oh seven"


class TestNormalization:
    def test_money_plain(self):
        assert "two hundred fifty dollars" in normalize_for_tts("It costs $250.")

    def test_money_with_unit(self):
        out = normalize_for_tts("A $1.2M grant was approved.")
        assert "one point two million dollars" in out

    def test_money_with_cents(self):
        assert "and fifty cents" in normalize_for_tts("$4.50 each")

    def test_percent(self):
        assert "twelve percent" in normalize_for_tts("Turnout rose 12%.")
        assert "three point five percent" in normalize_for_tts("Up 3.5% overall.")

    def test_time(self):
        out = normalize_for_tts("The meeting starts at 7:30pm.")
        assert "seven thirty P M" in out
        assert "7:30" not in out

    def test_abbreviations(self):
        out = normalize_for_tts("Dr. Reyes met Sen. Hall on Franklin Rd.")
        assert "Doctor Reyes" in out
        assert "Senator Hall" in out
        assert "Road" in out

    def test_acronyms_are_spelled_out(self):
        assert "A C M E" in normalize_for_tts("The ACME group reopened.")

    def test_protected_acronyms_stay_whole(self):
        assert "NASA" in normalize_for_tts("NASA confirmed the launch.")

    def test_custom_pronunciations_win(self):
        out = normalize_for_tts("the show meets tonight.", {"the show": "Post Six"})
        assert "Post Six" in out
        assert "Post six" not in out.replace("Post Six", "")

    def test_ampersand(self):
        assert " and " in normalize_for_tts("Smith & Jones")

    def test_empty_input(self):
        assert normalize_for_tts("") == ""


class TestChunking:
    def test_sentence_split(self):
        sentences = split_sentences("One thing. Then another! And a third?")
        assert len(sentences) == 3

    def test_abbreviation_does_not_split(self):
        sentences = split_sentences("Dr. Reyes spoke first. Then the vote came.")
        assert len(sentences) == 2
        assert sentences[0].startswith("Dr. Reyes")

    def test_chunks_respect_max_chars(self):
        text = " ".join(f"This is sentence number {i}." for i in range(40))
        chunks = chunk_text(text, max_chars=120)
        assert len(chunks) > 1
        assert all(len(c) <= 120 for c in chunks)

    def test_chunks_never_split_a_sentence(self):
        text = "Short one. " + "A " * 200 + "end."
        chunks = chunk_text(text, max_chars=100)
        # The oversized sentence is emitted whole rather than cut mid-word.
        assert any(len(c) > 100 for c in chunks)
        assert "".join(chunks).count("end.") == 1

    def test_chunking_preserves_all_text(self):
        text = "First sentence here. Second one follows. Third wraps it up."
        assert " ".join(chunk_text(text, max_chars=30)) == text

    def test_empty_text(self):
        assert chunk_text("") == []
