You receive:
1) audio in {language} (the source of truth),
2) REFERENCE Transcript {transcript}.

Goal: To normalize transcript refering to the audio.

CRITICALLY IMPORTANT (strict prohibitions):
- Do NOT “improve” or correct what was said: do not fix slips of the tongue, self-corrections, repetitions, or broken phrases if they are not present in the audio.
- Do NOT change entities. Keep names, places, brands, titles in exact written form as in REFERENCE Transcript. For entities follow only REFERENCE Transcript.

ALLOWED ONLY:
2) Add/fix punctuation and capitalization (this is formatting, not a change of meaning).
3) Normalize numeric expressions into words exactly as they are SPOKEN in the audio.
- Mixed format is forbidden:
    Bad: "5 percent", "2 zeros"
    Good: "five percent", "two zeros"
- Normalize: percentages, currencies, units, ranges, decimals, dates/years — ONLY if they are spoken.
- If a unit (for example “percent”) is NOT spoken, do not add it.

PUNCTUATION:
- Add punctuation marks (.,?!—:;) according to pauses and intonation in the audio.
- Do not restructure phrases for “smoothness”.

OUTPUT FORMAT:
- Return only the final text.
- No explanations, no JSON, no lists.

Input text {transcript}

