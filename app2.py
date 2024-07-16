from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.pydantic_v1 import BaseModel, Field
from quiz_templates import create_multiple_choice_template, create_true_false_template
from typing import List
import streamlit as st
import os
import random
from dotenv import load_dotenv 
from langchain.memory import ConversationBufferMemory

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

class QuizTrueFalse(BaseModel):
    quiz_text: str = Field(description="The quiz text")
    questions: List[str] = Field(description="The quiz questions")
    answers: List[str] = Field(description="The quiz answers for each question as True or False only.")

class QuizMultipleChoice(BaseModel):
    quiz_text: str = Field(description="The quiz text")
    questions: List[str] = Field(description="The quiz questions")
    alternatives: List[List[str]] = Field(description="The quiz alternatives for each question as a list of lists")
    answers: List[str] = Field(description="The quiz answers")

def create_quiz_chain(prompt_template, llm, pydantic_object_schema):
    """Creates the chain for the quiz app."""
    return prompt_template | llm.with_structured_output(pydantic_object_schema)

def split_questions_answers(quiz_response):
    """Function that splits the questions and answers from the quiz response."""
    questions = quiz_response.questions  # this will be a list of questions
    answers = quiz_response.answers  # this will be a list of answers
    return questions, answers

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    load_css("styles.css")  # Load custom CSS

    st.title("Quiz App")
    st.write("This app generates a quiz based on a given context.")

    # Initialize session state variables if they don't already exist
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.8)

    context = st.text_area("Enter the concept/context for the quiz", value=st.session_state.get('context', ''))
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=st.session_state.get('num_questions', 2))
    quiz_type = st.selectbox("Select the quiz type", ["multiple-choice", "true-false"], index=st.session_state.get('quiz_type_index', 0))

    if quiz_type == "multiple-choice":
        prompt_template = create_multiple_choice_template()
        pydantic_object_schema = QuizMultipleChoice
    elif quiz_type == "true-false":
        prompt_template = create_true_false_template()
        pydantic_object_schema = QuizTrueFalse

    if st.button("Generate Quiz"):
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.user_answers = []

        # Generate a random seed for each quiz generation to ensure randomness
        random_seed = random.randint(1, 10000)
        st.session_state.random_seed = random_seed
        chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)
        quiz_response = chain.invoke({"num_questions": num_questions, "quiz_context": context, "random_seed": random_seed})
        st.write(quiz_response)
        st.session_state.questions = quiz_response.questions
        st.session_state.answers = quiz_response.answers
        if quiz_type == "multiple-choice":
            st.session_state.alternatives = quiz_response.alternatives
        st.session_state.user_answers = [None] * len(quiz_response.questions)
        st.session_state.context = context
        st.session_state.num_questions = num_questions
        st.session_state.quiz_type_index = ["multiple-choice", "true-false"].index(quiz_type)

    if st.session_state.questions:
        display_questions(quiz_type)
        if st.button("Submit Answers"):
            process_submission(quiz_type)

def display_questions(quiz_type):
    if quiz_type == "multiple-choice":
        for i, question in enumerate(st.session_state.questions):
            st.markdown(question)
            options = st.session_state.alternatives[i]  # Descriptive options
           
            selected_option = st.radio("Select your answer", options, key=f"question_{i}", index=None)
            
            option_index = options.index(selected_option) if selected_option is not None else None
            option_identifier = chr(97 + option_index) if option_index is not None else None
            st.session_state.user_answers[i] = option_identifier
    
    elif quiz_type == "true-false":
        for i, question in enumerate(st.session_state.questions):
            st.markdown(question)
            
            selected_option = st.radio("Select your answer", ["True", "False"], key=f"question_{i}", index=None)
            st.session_state.user_answers[i] = selected_option

def process_submission(quiz_type):
    if 'user_answers' in st.session_state:
        if None in st.session_state.user_answers:
            st.warning("Please answer all the questions before submitting.")
        else:
            correct_icon = "✅"  # Green tick mark
            incorrect_icon = "❌"  # Red cross mark
            score = 0
            
            st.write("Quiz Results:")
            for i, question in enumerate(st.session_state.questions):
                st.markdown(question)
                user_answer = st.session_state.user_answers[i]
                correct_answer = st.session_state.answers[i]
                
                if quiz_type == "multiple-choice":
                    # Use user_answer directly as it's already an option identifier ('A', 'B', 'C', ...)
                    selected_option = st.session_state.alternatives[i][ord(user_answer) - 97] if user_answer is not None else None
                else:
                    selected_option = user_answer
                
                if user_answer == correct_answer:
                    st.write(f"Your answer: {selected_option} {correct_icon}")
                    score += 1
                else:
                    if quiz_type == "multiple-choice":
                        correct_option = st.session_state.alternatives[i][ord(correct_answer) - 97] if correct_answer is not None else None
                    else:
                        correct_option = correct_answer
                    st.write(f"Your answer: {selected_option} {incorrect_icon} | Correct answer: {correct_option} {correct_icon}")
            
            st.write(f'Your score is {score}/{len(st.session_state.questions)}')

if __name__ == "__main__":
    main()
