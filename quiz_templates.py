# quiz_templates.py
from langchain.prompts import ChatPromptTemplate

def create_multiple_choice_template():
    template = """
You are an expert quiz maker for technical fields. Create a multiple-choice quiz with exactly {num_questions} questions about: {quiz_context}.

Your response must strictly follow this format:
1. A quiz_text field with a short title for the quiz
2. A questions field with a list of {num_questions} clear, concise questions
3. An alternatives field with exactly 4 options for each question (without a, b, c, d prefixes in the text)
4. An answers field with exactly {num_questions} correct answers marked as a, b, c, or d

Example structure:
{{
  "quiz_text": "Basic Python Quiz",
  "questions": ["What keyword is used to define a function in Python?", "Which of these is not a Python data type?"],
  "alternatives": [["def", "function", "define", "func"], ["list", "dictionary", "array", "tuple"]],
  "answers": ["a", "c"]
}}

Make sure each question has exactly 4 alternatives, and answers are only single letters (a, b, c, or d).
DO NOT include the a., b., c., d. prefixes in the alternatives text.
Random seed: {random_seed}
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_true_false_template():
    template = """
You are an expert quiz maker for technical fields. Create a true-false quiz with exactly {num_questions} questions about: {quiz_context}.

Your response must strictly follow this format:
1. A quiz_text field with a short title for the quiz
2. A questions field with a list of {num_questions} clear true/false statements
3. An answers field with exactly {num_questions} correct answers marked as either "True" or "False"

Example structure:
{{
  "quiz_text": "Basic Python Facts",
  "questions": ["Python is a statically typed language.", "Python was first released in 1991."],
  "answers": ["False", "True"]
}}

Make sure answers are exactly "True" or "False" strings.
Random seed: {random_seed}
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt