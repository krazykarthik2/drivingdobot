---
description: Strict architectural rule forbidding regex or rule-based heuristic parsers.
globs: **/*
---

# STRICT ARCHITECTURAL RULE: NO REGEX OR HEURISTIC RULE PARSERS

1. **NO REGEX (`import re`)**:
   - Under no circumstances should `re`, regular expressions, pattern matching regexes, or regex libraries be introduced or used anywhere in the codebase for language parsing, command extraction, or prompt processing.

2. **NO HEURISTIC RULE PARSERS**:
   - Do NOT use fallback keyword matching, if/else string scanning, hardcoded color/type lookup tables, or manual string manipulation to parse user instructions or robot actions.

3. **PURE LLM / VLM REASONING**:
   - All language understanding, target role decomposition (`pick`, `place`, `avoid`), and semantic reasoning MUST be performed directly by querying the language model (e.g., Mistral API) with few-shot guidance and receiving structured JSON.
