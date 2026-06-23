You receive {language} audio and a reference transcript. The reference may be cleaned, partially wrong, or missing speech artifacts. The audio is the ground truth.

REFERENCE TRANSCRIPT:
"{transcript}"

TASK: Revise the reference to match exactly what is spoken in {language}, including all disfluencies.
- Use the reference as a starting point; keep parts that already match the audio.
- When reference and audio conflict, follow the audio.
- Do NOT invent, remove, paraphrase, or polish content. Prefer minimal edits.
- Keep text in languages other than {language} as is; do not correct it. This can be code-switched data.

ENTITIES: Keep every named entity from the reference in its exact written form (spelling, casing, script, punctuation). Do not transliterate, translate, or re-spell unless the audio clearly shows a different form. Preserve code-switched entities unchanged.

SPEECH ARTIFACTS: Keep and add fillers, repetitions, stutters (e.g. "I I think"), false starts (incomplete words/phrases marked with a hyphen), colloquial/informal forms, and grammatical errors as spoken. Do NOT clean up, normalize, or expand informal speech.
 
NUMBERS: Write numbers as spoken in words in {language}.

OUTPUT: Return ONLY the revised transcription text—no explanations, JSON, or lists.
