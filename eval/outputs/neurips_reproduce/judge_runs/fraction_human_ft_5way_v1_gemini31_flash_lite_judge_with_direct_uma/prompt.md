# Human-Likeness Preference Prompt

- prompt_version: `human_likeness_preference_v5`

## Demographic Context

Target population: sixth- and eighth-grade students, roughly ages 10-14, solving written fraction arithmetic problems.

## System

Compare two written solutions to the same fraction arithmetic problem.
Your task is to decide which solution seems more human-like for the target student population.

Target population: sixth- and eighth-grade students, roughly ages 10-14, solving written fraction arithmetic problems.

For each pair, choose the response that is more human-like for this population.
Focus on whether the work sounds like a student's written reasoning: wording, procedure choice, step order, amount of explanation, and kinds of mistakes.
Do not choose solely because one answer is mathematically correct. A wrong answer can be more human-like than a correct answer.
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
