#import libs

import streamlit as st
import os
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import openai
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
import warnings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uuid
from datetime import datetime, timedelta



# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Islamic Chatbot", layout="wide")
st.title("Islamic Chatbot")
st.write("Hello! I'm an Islamic chatbot. I'm here to provide information and guidance on Islamic topics. Let's start!")

# Load environment variables from .env file
#load_dotenv()


#For Streamlit & AWS
#OpenAI API key
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
#Groq API KEY
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]

# For Heroku & Local deployment
#OPENAI_API_KEY = os.getenv("My_OpenAI_API_key")
#GROQ_API_KEY = os.getenv("My_Groq_API_key")

# Model selection
model_options = ["llama3-70b-8192", "llama3-8b-8192","gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select a model", model_options)

# Initialize selected model
def get_model(selected_model):
    if selected_model == "llama3-70b-8192":
        return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2) 
    elif selected_model == "llama3-8b-8192":
        return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
    elif selected_model ==  "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    elif selected_model == "gpt-4":
        return ChatOpenAI(model="gpt-4", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Invalid model selected")

llm_mod = get_model(selected_model)

system_prompt = """
You are an Islamic Chatbot. Your primary tasks involve providing information and guidance based on Islamic teachings. Follow these rules for each task:

1. **Islamic Information**:
   - Provide information and guidance based on Islamic teachings, including Quran, Hadith, and scholarly interpretations.
   - Ensure that your responses are aligned with the principles of Islam.

2. **Handling Haram Inquiries**:
   - If a user asks a question that involves haram (forbidden) topics, respond with:
     "I'm an Islamic Chatbot and can't respond to haram inquiries."

3. **General Islamic Guidance**:
   - Offer guidance on daily prayers, fasting, zakat, Hajj, and other pillars of Islam.
   - Provide tips on living an Islamic lifestyle and following the teachings of Islam in daily life.

4. **Summary Confirmation**:
   - Display a summary of the information or guidance provided to ensure the user understands the response.
   - Confirm with the user if they need further assistance or have additional questions.

5. **Completion**:
   - Upon providing the necessary information and guidance, offer a confirmation to the user, including next steps or follow-up actions if needed.

6. **Off-topic Handling**:
   - If the user asks a question that is not related to Islamic teachings, respond with:
     "Sorry, but I'm here to assist you with information and guidance based on Islamic teachings. If you have any questions related to these topics, please feel free to ask!"

7. **Capability Handling**:
   - If the user asks a question about your capabilities or functions, respond with:
     "Sorry, I was trained to assist with information and guidance based on Islamic teachings. If you have any questions related to these topics, please feel free to ask!"

8. **Capability Confirmation**:
   - If the user wants to confirm whether you're trained specifically for Islamic guidance, respond with:
     "Yes, I was trained to assist with only Islamic teachings and guidance. If you have any questions related to these topics, please feel free to ask!"

You must follow this rule for handling multiple function calls in a single message:

1. For any "create" function (e.g., creating a list of Islamic resources), you must first summarize the data and present it to the user for confirmation.
2. Only after the user confirms the correctness of the data should you proceed to submit the function call.

Here's how you should handle it:
• Summarize the data in a clear and concise manner.
• Ask the user for confirmation with a clear question, e.g., "Do you confirm the above data? (Yes/No)"
• If the user confirms, proceed to make the function call.
• If the user does not confirm or requests changes, modify the data as per the user's instructions and present it again for confirmation.
"""

# Initialize the conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'sessions' not in st.session_state:
    st.session_state['sessions'] = {}

if 'current_session_name' not in st.session_state:
    st.session_state['current_session_name'] = None

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

if 'conversation_state' not in st.session_state:
    st.session_state['conversation_state'] = "start"

# Function to get today's date in a readable format
def get_readable_date():
    return datetime.now().strftime("%Y-%m-%d")

# Function to generate a summary of the user's first query
def generate_summary(user_input):
    summary_prompt = f"Summarize this query in a few words: {user_input}"
    summary_response = llm_mod.predict(summary_prompt)
    return summary_response.strip()

# Function to generate a unique session name based on the summary of the user's first query
def generate_session_name(user_input):
    summary = generate_summary(user_input)
    return summary

# Function to save the current session
def save_current_session():
    if st.session_state['current_session_name'] and len(st.session_state['messages']) > 1:
        st.session_state['sessions'][st.session_state['current_session_name']] = {
            'date': get_readable_date(),
            'messages': st.session_state['messages'].copy()
        }

# Function to display chat sessions in the sidebar
def display_chat_sessions():
    st.sidebar.header("Chat History")
    today = get_readable_date()
    yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    sessions = sorted(st.session_state['sessions'].items(), key=lambda x: x[1]['date'], reverse=True)
    
    current_day = ""
    for session_name, session_info in sessions:
        session_day = session_info['date']
        if session_day != current_day:
            if session_day == today:
                st.sidebar.subheader("Today")
            elif session_day == yesterday:
                st.sidebar.subheader("Yesterday")
            else:
                st.sidebar.subheader(session_day)
            current_day = session_day
        if st.sidebar.button(session_name):
            st.session_state['messages'] = session_info['messages']

# Display saved chat sessions in the sidebar
display_chat_sessions()

# Display chat messages from history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def clear_input():
    st.session_state.user_input = ''

if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

user_question = st.chat_input("You: ")

if user_question:
    st.session_state.user_input = user_question

    # Set session name based on the summary of the first user input
    if st.session_state.current_session_name is None:
        st.session_state.current_session_name = generate_session_name(st.session_state.user_input)
    
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": st.session_state.user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(st.session_state.user_input)
        
    if st.session_state.conversation_state == "start":
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ])

    elif st.session_state.conversation_state == "awaiting_confirmation":
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
            HumanMessagePromptTemplate.from_template("The user has confirmed the data. Proceed with providing guidance."),
        ])
        
    conversation = LLMChain(
        llm=llm_mod,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    with st.spinner("Thinking..."):
        try:
            response = conversation.predict(human_input=st.session_state.user_input)
            if "Do you confirm the above data?" in response:
                st.session_state.conversation_state = "awaiting_confirmation"
            elif "Proceeding with detailed guidance" in response:
                st.session_state.conversation_state = "providing_guidance"
            else:
                st.session_state.conversation_state = "start"
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            response = "Sorry, I'm having trouble processing your request right now. Please try again later."

    # Add bot response to chat history
    st.session_state['messages'].append({"role": "assistant", "content": response})
    
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    clear_input() # Clear the input field
    
    # Save the current session automatically
    save_current_session()
