"""Prompts. `tehlar` is the text Tehlar's main.py actually sent (mistral_prompt): Python joined the
backslash-continued lines, so the leading spaces and the '#' markers are part of it."""

TEHLAR = (
    "You are an IQ test solver. You are presented with an image of an IQ test question. The IQ test question is a"
    "    3x3 grid with nine boxes. The bottom-right box is empty and marked with a symbol 'x'. Following the instructions, you are to"
    "    deduce the pattern that links the shapes in the matrix and select the correct answer from the options provided under the matrix,"
    "    separated by a thick horizontal line, labeled A to H."
    "    ### Instructions:"
    "    #1. Carefully observe the changes across each row, column, and diagonals, noting any consistent transformations (e.g., shape changes, rotations, or color shifts) that might follow a sequence or logical rule."
    "    # 2. Look for common pattern types like rotation, mirroring, color shades, or element counting, explaining briefly why each might apply."
    "    # 3. Identify any unique features in the shapes."
    "    # 4. Eliminate options (A to H) that clearly do not match the observed patterns."
    "    # 5. Select the correct shape that completes the pattern."
    "    # Take your time with each step and be thorough in your reasoning. Describe the thinking process."
    "    # Conclude with:"
    "    'The correct answer is: _'."
)

TEHLAR_CLEAN = """You are an IQ test solver. You are presented with an image of an IQ test question. The question is a 3x3 grid with nine boxes. The bottom-right box is empty and marked with an 'x'. Deduce the pattern that links the shapes in the matrix and select the correct answer from the options under the matrix, separated by a thick horizontal line and labeled A to H.

Instructions:
1. Carefully observe the changes across each row, column and diagonal, noting any consistent transformations (e.g. shape changes, rotations or colour shifts) that might follow a sequence or logical rule.
2. Look for common pattern types like rotation, mirroring, colour shades or element counting, explaining briefly why each might apply.
3. Identify any unique features in the shapes.
4. Eliminate options (A to H) that clearly do not match the observed patterns.
5. Select the correct shape that completes the pattern.

Take your time with each step and be thorough in your reasoning. Describe the thinking process.
Conclude with exactly one line: 'The correct answer is: X', where X is a single letter from A to H."""

XML_TEXT = """You are an IQ test solver. Below is a Raven's Progressive Matrices question written out as text: a 3x3 matrix of panels with the bottom-right panel missing, and eight candidate answers labelled A to H. Each panel lists its objects with shape, size (0 = smallest, 5 = largest) and colour (0 = white, 9 = black).

Deduce the rules that link the panels across each row and select the candidate that completes the matrix. Explain your reasoning, then conclude with exactly one line: 'The correct answer is: X', where X is a single letter from A to H.

{puzzle}"""

IRAVENX_TEXT = """You are an IQ test solver. Below is a Raven's Progressive Matrices question written out as text: rows of a matrix of panels with the last panel of the last row missing, and eight candidate answers labelled A to H. Each panel is one object written as (type, size, colour), each a non-negative integer.

Deduce the rules that link the panels across each row and select the candidate that completes the matrix. Explain your reasoning, then conclude with exactly one line: 'The correct answer is: X', where X is a single letter from A to H.

{puzzle}"""

PROMPTS = {"tehlar": TEHLAR, "tehlar_clean": TEHLAR_CLEAN, "xml_text": XML_TEXT, "iravenx_text": IRAVENX_TEXT}
TEXT_PROMPTS = {"xml_text", "iravenx_text"}
