You receive {language} audio and a reference transcript. The reference may be cleaned, partially wrong, or missing speech artifacts. The audio is the ground truth.

REFERENCE TRANSCRIPT:
"{transcript}"

MAIN GOAL: Listen carefully to the audio and revise the reference so it faithfully reflects exactly what is spoken in {language}, including all disfluencies present in the audio.
- Use the reference as a starting point; do not ignore it.
- When the reference matches the audio, keep it unchanged.
- When the reference conflicts with the audio, follow the audio.
- Do NOT invent words or content not spoken in the audio.
- Do NOT remove substantive content that is spoken in the audio (remove reference words only if they are not spoken).
- Do NOT paraphrase, polish grammar, or rewrite sentences that already match the audio.
- Prefer minimal edits: fix mismatches and insert missing speech artifacts.
- Preserve named entities from the reference in their exact written form
- Normalize numbers to their spoken form in their source language {language}.
- Keep code-switched text as is.


ENTITIES (names, places, brands, titles, etc.):
- Keep every named entity from the reference in its exact written form: spelling, casing, script, and punctuation. This includes names, places, brands, titles, acronyms, and other proper nouns.
- Do not transliterate, translate, re-spell, normalize, or "correct" an entity into another script or language unless the audio clearly shows a different entity or form was spoken.
- If enetities are part code switched data it should stay the same.

KEEP REFERENCE DISFLUENCIES:
- If the reference already has fillers, repetitions, false starts, colloquial reductions, or grammatical errors, keep them.
- Add hesitation markers and fillers natural to {language} wherever they are spoken in the audio but missing from the reference.
- Do NOT clean up, normalize, or remove disfluencies that are already in the reference and are spoken in the audio.
- Add consecutive instances of the same word or short phrase when spoken unintentionally.
  - Example: reference "I think" → "I I think" if that is what is spoken.

FALSE STARTS:
- Add incomplete words or phrases the speaker abandons, marked with a hyphen.
- Do NOT remove false starts already in the reference if they are spoken in the audio.

COLLOQUIAL / INFORMAL FORMS:
- If the reference uses a standard or formal form but the speaker used a colloquial, reduced, or informal form in {language}, use the spoken form.
- Preserve colloquial and informal forms exactly as spoken. Do NOT expand them into standard or formal written forms.

WRONG GRAMMAR:
- Keep grammatical errors as spoken. Do NOT correct grammar, agreement, tense, or other linguistic errors.

NUMERICALS:
- Keep numbers as spoken in words in {language}. Do NOT convert them to digits unless that is how they were spoken.

Output format:
- Return ONLY the revised transcription text.
- No explanations, no JSON, no lists.

