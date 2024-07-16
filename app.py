from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

load_dotenv()

def create_the_quiz_prompt_template():
    """Create the  prompt template for the quiz."""
    template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with {num_questions} {quiz_type} questions about the following concept/content: {quiz_context}.

The format of the quiz could be one of the following:
- Multiple-choice: 
- Questions:
    <Question1>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    <Question2>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
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
        - 1. What is a binary search tree?
        - 2. How are binary search trees implemented?
    - Answers:
        - 1. True
        - 2. False
- Open-ended:
- Questions:
    <Question1>: 
    <Question2>:
- Answers:    
    <Answer1>:
    <Answer2>:
Example:
    Questions:
    - 1. What is a binary search tree?
    - 2. How are binary search trees implemented?
    
    - Answers: 
        1. A binary search tree is a data structure that is used to store data in a sorted manner.
        2. Binary search trees are implemented using linked lists.
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_quiz_chain(prompt_template,llm):
    """Create the chain for the quiz app."""
    return prompt_template | llm | StrOutputParser()

def split_questions_answers(quiz_response):
    """Function that split the questions and answers from the quiz response."""
    if "Answers:" not in quiz_response:
        return quiz_response, "No answers provided."
    questions = quiz_response.split("Answers:")[0].strip()
    answers = quiz_response.split("Answers:")[1].strip()
    return questions, answers

def main():
    st.title("Quiz App")
    st.write("Welcome to the Quiz")
    prompt_template = create_the_quiz_prompt_template()
    llm = ChatGroq(model="mixtral-8x7b-32768")
    chain = create_quiz_chain(prompt_template,llm)
    context = st.text_area("Enter the context/concept of the quiz")
    num_questions = st.number_input("Enter the number of questions",min_value=1,max_value=10,value=3)
    quiz_type = st.selectbox("Select the type of quiz",["Multiple_choice","True-False","Open-ended"])

    if st.button("Generate_Quiz"):
        quiz_response = chain.invoke({"quiz_type":quiz_type,"num_questions":num_questions,"quiz_context":context})
        st.write("Quiz Generated !")
        questions,answers = split_questions_answers(quiz_response)
        st.session_state.answers = answers
        st.session_state.questions = questions
        st.write(questions)  
    if st.button("Show Answers"):
        st.markdown(st.session_state.questions)
        st.write("----")
        st.markdown(st.session_state.answers)

if __name__ == "__main__":
    main()