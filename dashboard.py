import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os


load_dotenv()


groq_model = Groq(id="llama3-70b-8192")  # ✅ Valid model ID


web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for general information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources in your response."],
    show_tools_calls=True,
    markdown=True,
)

financial_agent = Agent(
    name="Financial Agent",
    role="Answer financial questions about stocks and companies",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
        DuckDuckGo()
    ],
    instructions=[
        "Use tables to display stock data.",
        "Always include sources in your response."
    ],
    show_tools_calls=True,
    markdown=True,
)


st.set_page_config(page_title="AgentiAI", layout="centered")
st.title("🧠 AgentiAI – Multi-Agent Chatbot")


agent_choice = st.selectbox("🧠 Choose your Agent", ["Web Search Agent", "Financial Agent"])
agent = web_search_agent if agent_choice == "Web Search Agent" else financial_agent


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"], avatar=chat["avatar"]):
        st.markdown(chat["message"], unsafe_allow_html=True)

user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    
    st.chat_message("user", avatar="🧑").markdown(user_prompt)
    st.session_state.chat_history.append({
        "role": "user",
        "message": user_prompt,
        "avatar": "🧑"
    })

    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = agent.run(user_prompt)
                response = result.content
            except Exception as e:
                response = f" Error: {str(e)}"
        st.markdown(response, unsafe_allow_html=True)

    
    st.session_state.chat_history.append({
        "role": "assistant",
        "message": response,
        "avatar": "🤖"
    })
