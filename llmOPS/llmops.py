from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.chat_models import ChatOpenAI

import streamlit as st
import os
#from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = "trweqT3BlbkFJEP7EaE-1IAYA"
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langmith tracking
os.environ["LANGSMITH_TRACING_V2"]="true"

os.environ["LANGSMITH_API_KEY"]= "l1db6ab"
#os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","I am chatbot. I am hear to assist you. Please type your queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('LLM-OPENAI PROJECT - CUSTOM GPT-5 BY Ravi Teja Madabathula')
input_text=st.text_input("How may I help you")

# openAI LLm
llm=ChatOpenAI(model="gpt-5.1", temperature=0.2)
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))