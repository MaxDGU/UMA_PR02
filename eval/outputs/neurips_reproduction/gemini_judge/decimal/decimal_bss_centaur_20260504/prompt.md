# Human-Likeness Preference Prompt

- prompt_version: `human_likeness_preference_v5`
- plan: `decimal_bss_anchor_v1`

## Demographic Context

Target population: sixth- and eighth-grade students, roughly ages 10-14, solving written decimal arithmetic problems.

## Format Note

Some responses may be compact transcriptions of written work, while others may be prose explanations. Do not prefer a response just because it is longer, more polished, or written in complete sentences.

## System

Compare two written solutions to the same decimal arithmetic problem.
Your task is to decide which solution seems more human-like for the target student population.

Target population: sixth- and eighth-grade students, roughly ages 10-14, solving written decimal arithmetic problems.

For each pair, choose the response that is more human-like for this population.
Focus on whether the work sounds like a student's written reasoning: wording, procedure choice, step order, amount of explanation, and kinds of mistakes.
Do not choose solely because one answer is mathematically correct. A wrong answer can be more human-like than a correct answer.
Some responses may be compact transcriptions of written work, while others may be prose explanations. Do not prefer a response just because it is longer, more polished, or written in complete sentences.
Do not assume either response is human-written or model-generated.
Return JSON only.

## User Template

Problem: {problem}

Response A:
{response_a}

Response B:
{response_b}

Which response is more human-like for the target student population?

Return exactly one JSON object with this schema:
{"preferred": "A|B|TIE", "confidence": "low|medium|high", "reason": "one short sentence"}
