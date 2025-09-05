import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

#  AI Agent Logic 
def get_response_from_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    else:
        return {"error": "Invalid provider"}

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        messages_modifier=system_prompt
    )

    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    if not ai_messages:
        return {"response": "⚠️ No response from agent"}
    return {"response": ai_messages[-1]}

#  frontend part(streamlit)
st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agents")
st.write("Create and interact with the AI agents!")

# Session State
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# System prompt input
st.session_state.system_prompt = st.text_area(
    "Define your AI Agent:",
    height=70,
    placeholder="Type your system prompt here...",
    value=st.session_state.system_prompt
)

# Model Selection
MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))
selected_model = st.selectbox(
    "Select Model:",
    MODEL_NAMES_GROQ if provider == "Groq" else MODEL_NAMES_OPENAI
)

# Search option
allow_web_search = st.checkbox("Allow Web Search")

# User query
st.session_state.user_query = st.text_area(
    "Enter your query:",
    height=150,
    placeholder="Ask your AI agent...",
    value=st.session_state.user_query
)

# Run Agent Button
if st.button("Ask Agent!"):
    if st.session_state.user_query.strip():
        with st.spinner("🤔 Thinking..."):
            result = get_response_from_agent(
                llm_id=selected_model,
                query=st.session_state.user_query,
                allow_search=allow_web_search,
                system_prompt=st.session_state.system_prompt or "You are a helpful AI agent.",
                provider=provider
            )
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Agent Response")
            st.markdown(result["response"].replace("\n", "<br>"), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a query to ask the agent.")