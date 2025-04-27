# app_test.py
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
import json

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

os.environ["GROQ_API_KEY"] = groq_api_key


class QuizTrueFalse(BaseModel):
    quiz_text: str = Field(description="The quiz text")
    questions: List[str] = Field(description="The quiz questions")
    answers: List[str] = Field(description="The quiz answers for each question as True or False only.")


class QuizMultipleChoice(BaseModel):
    quiz_text: str = Field(description="The quiz title or topic")
    questions: List[str] = Field(description="List of quiz questions")
    alternatives: List[List[str]] = Field(description="List of answer choices for each question")
    answers: List[str] = Field(description="Correct answer for each question as a, b, c or d")


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
    try:
        load_css("styles.css")
    except FileNotFoundError:
        st.warning("styles.css file not found. Using default styling.")
    
    st.title("Quiz App")
    st.write("This app generates a quiz based on a given context.")

    # Initialize session state variables if they don't already exist
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if 'submitted_quizzes' not in st.session_state:
        st.session_state.submitted_quizzes = []
    if 'selected_quiz' not in st.session_state:
        st.session_state.selected_quiz = None

    # Changed to a more powerful model
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)
    memory = st.session_state.memory

    context = st.text_area("Enter the concept/context for the quiz", value=st.session_state.get('context', ''))
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=st.session_state.get('num_questions', 3))
    quiz_type = st.selectbox("Select the quiz type", ["multiple-choice", "true-false"], index=st.session_state.get('quiz_type_index', 0))

    if quiz_type == "multiple-choice":
        prompt_template = create_multiple_choice_template()
        pydantic_object_schema = QuizMultipleChoice
    elif quiz_type == "true-false":
        prompt_template = create_true_false_template()
        pydantic_object_schema = QuizTrueFalse

    if st.button("Generate Quiz"):
        if not context:
            st.error("Please enter a context for the quiz.")
            return
            
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.user_answers = []

        # Generate a random seed for each quiz generation to ensure randomness
        random_seed = random.randint(1, 10000)
        st.session_state.random_seed = random_seed
        
        with st.spinner("Generating quiz... This may take a moment."):
            try:
                chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)

                # Load previous memory
                previous_memory = memory.load_memory_variables(inputs={"context": context})
                previous_memory_str = previous_memory.get("context", "")
                # Update context with previous memory
                context_with_memory = f"{context}\n{previous_memory_str}" if previous_memory_str else context

                quiz_response = chain.invoke({
                    "num_questions": str(num_questions),
                    "quiz_context": context_with_memory,
                    "random_seed": str(random_seed)
                })
                
                st.session_state.questions = quiz_response.questions
                st.session_state.answers = quiz_response.answers
                if quiz_type == "multiple-choice":
                    st.session_state.alternatives = quiz_response.alternatives
                st.session_state.user_answers = [None] * len(quiz_response.questions)
                st.session_state.context = context
                st.session_state.num_questions = num_questions
                st.session_state.quiz_type_index = ["multiple-choice", "true-false"].index(quiz_type)

                # Save current context to memory - convert to string first
                memory.save_context(
                    inputs={"context": context}, 
                    outputs={"response": json.dumps(quiz_response.dict(), default=str)}
                )
                
                st.success("Quiz generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred while generating the quiz: {e}")
                st.write(f"Error details: {str(e)}")

    if st.session_state.questions:
        display_questions(quiz_type)
        if st.button("Submit Answers"):
            process_submission(quiz_type)

    # Display memory on the sidebar
    st.sidebar.title("Quiz History")
    if 'memory' in st.session_state:
        memory_content = st.session_state.memory.buffer
        st.sidebar.write(memory_content)
    
    # Display submitted quizzes on the sidebar
    st.sidebar.title("Submitted Quizzes")
    for i, quiz in enumerate(st.session_state.submitted_quizzes):
        if st.sidebar.button(f"{quiz['title']}", key=f"quiz_{i}"):
            st.session_state.selected_quiz = i

    # Display the selected quiz details
    if st.session_state.selected_quiz is not None:
        selected_quiz = st.session_state.submitted_quizzes[st.session_state.selected_quiz]
        st.write(f"Quiz Title: {selected_quiz['title']}")
        for question, answer, user_answer in zip(selected_quiz['questions'], selected_quiz['answers'], selected_quiz['user_answers']):
            st.write(f"Q: {question}")
            st.write(f"Correct Answer: {answer}")
            st.write(f"Your Answer: {user_answer}")
        st.session_state.questions = selected_quiz['questions']  # Ensure questions are updated
        st.session_state.answers = selected_quiz['answers']  # Ensure answers are updated
        if selected_quiz['alternatives']:
            st.session_state.alternatives = selected_quiz['alternatives']  # Ensure alternatives are updated if multiple-choice


def display_questions(quiz_type):
    if quiz_type == "multiple-choice":
        for i, question in enumerate(st.session_state.questions):
            st.markdown(f"**Q{i+1}: {question}**")
            options = st.session_state.alternatives[i]

            # दोहराव हटाने के लिए
            labeled_options = [f"{chr(97 + j)}. {option}" for j, option in enumerate(options)]

            selected_option = st.radio("Select your answer:", labeled_options, key=f"question_{i}", index=None)

            if selected_option:
                option_identifier = selected_option[0]  # पहला अक्षर (a, b, c, d)
                st.session_state.user_answers[i] = option_identifier
            else:
                st.session_state.user_answers[i] = None

    elif quiz_type == "true-false":
        for i, question in enumerate(st.session_state.questions):
            st.markdown(f"**Q{i+1}: {question}**")
            selected_option = st.radio("Select your answer:", ["True", "False"], key=f"question_{i}", index=None)
            st.session_state.user_answers[i] = selected_option


def process_submission(quiz_type):
    if 'user_answers' in st.session_state:
        if None in st.session_state.user_answers:
            st.warning("Please answer all the questions before submitting.")
        else:
            correct_icon = "✅"  # Green tick mark
            incorrect_icon = "❌"  # Red cross mark
            score = 0

            st.write("## Quiz Results:")
            for i, question in enumerate(st.session_state.questions):
                st.markdown(f"**Q{i+1}: {question}**")
                user_answer = st.session_state.user_answers[i]
                correct_answer = st.session_state.answers[i]

                if quiz_type == "multiple-choice":
                    # Safety check for valid index
                    try:
                        selected_option = st.session_state.alternatives[i][ord(user_answer) - 97] if user_answer is not None else "No answer"
                    except (IndexError, TypeError):
                        selected_option = "Error: Invalid option"
                else:
                    selected_option = user_answer

                if user_answer == correct_answer:
                    st.write(f"Your answer: {selected_option} {correct_icon}")
                    score += 1
                else:
                    if quiz_type == "multiple-choice":
                        # Safety check for valid index
                        try:
                            correct_option = st.session_state.alternatives[i][ord(correct_answer) - 97] if correct_answer is not None else "Unknown"
                        except (IndexError, TypeError):
                            correct_option = "Error: Invalid option"
                    else:
                        correct_option = correct_answer
                    st.write(f"Your answer: {selected_option} {incorrect_icon} | Correct answer: {correct_option} {correct_icon}")

            st.write(f'### Your score is {score}/{len(st.session_state.questions)}')

            # Save the submitted quiz
            st.session_state.submitted_quizzes.append({
                "title": st.session_state.context,
                "questions": st.session_state.questions,
                "answers": st.session_state.answers,
                "user_answers": st.session_state.user_answers,
                "alternatives": st.session_state.alternatives if quiz_type == "multiple-choice" else None  # Save alternatives if multiple-choice
            })


if __name__ == "__main__":
    main()