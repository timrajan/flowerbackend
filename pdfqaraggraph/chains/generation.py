from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context. 

Instructions:
- Use ONLY the information provided in the context below to answer the question
- If the answer is not found in the context, say "I cannot find the answer in the provided context"
- Be accurate and concise in your response

Context:
{context}

Question: {question}

Answer:""")
load_dotenv()
llm = ChatOpenAI(temperature=0)

# Your existing chain works the same
generation_chain = prompt | llm | StrOutputParser()