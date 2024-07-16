from langchain.prompts import ChatPromptTemplate

def create_multiple_choice_template():
   
    template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with {num_questions}  questions about the following concept/content: {quiz_context}.

The format of the quiz as is follows:
- Multiple-choice: 
- Questions:
    <Question1>:
      -Alternatives1: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    <Question2>: 
      -Alternatives2: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    ....


- Answers:
    <Answer1>: <a|b|c|d>
    <Answer2>: <a|b|c|d>
    ....
    Example:
    - Questions:
    - 1. What is the time complexity of a binary search tree?
        a. O(n)
        b. O(log n)
        c. O(n^2)
        d. O(1)
    - Answers: 
        1. b
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_true_false_template():
    template = """
    You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with {num_questions} true-false questions about the following concept/content: {quiz_context}.

The format of the quiz is as follows:
- True-false:
- Questions:
    <Question1>: <True|False>
    <Question2>: <True|False>
    .....
- Answers:
    <Answer1>: <True|False>
    <Answer2>: <True|False>
    .....
Example:
- Questions:
    - 1. Is a binary search tree a type of data structure?
    - 2. Do binary search trees store data in an unsorted manner?
- Answers:
    - 1. True
    - 2. False
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

