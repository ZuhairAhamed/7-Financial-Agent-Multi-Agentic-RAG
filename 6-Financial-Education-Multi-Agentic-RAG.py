"""
Personalized Financial Education Multi-Agentic RAG System
"""
import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USER_AGENT"] = "FinEduMultiAgentBot/1.0"

import numpy as np
from dotenv import load_dotenv
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain import hub
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START

#load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# --- Define domain keywords for each agent ---
domain_keywords = {
    "budgeting_agent": ["budget", "budgeting", "expense", "saving", "spending", "track", "plan", "save", "money management"],
    "investment_agent": ["invest", "investment", "stock", "mutual fund", "etf", "portfolio", "risk", "diversification", "returns"],
    "credit_agent": ["credit score", "credit report", "improve credit", "fico", "credit history", "credit card", "loan approval"],
    "regulation_agent": ["regulation", "consumer rights", "protection", "law", "finance law", "financial regulation", "compliance"],
}

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
domain_embeddings = {
    agent: embedding_model.embed_documents(keywords)
    for agent, keywords in domain_keywords.items()
}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def setup_vector_stores():
    """Set up vector stores for each financial education domain."""
    budgeting_urls = [
        "https://www.investopedia.com/terms/b/budget.asp",
        "https://www.nerdwallet.com/article/finance/how-to-budget"
    ]
    investment_urls = [
        "https://www.investopedia.com/terms/i/investing.asp",
        "https://www.nerdwallet.com/article/investing/investing-101"
    ]
    credit_urls = [
        "https://www.investopedia.com/terms/c/credit_score.asp",
        "https://www.experian.com/blogs/ask-experian/credit-education/"
    ]
    regulation_urls = [
        "https://www.consumerfinance.gov/rules-policy/regulations/",
        "https://www.investopedia.com/terms/f/financial-regulation.asp"
    ]

    def load_and_split(urls):
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(docs_list)

    budgeting_splits = load_and_split(budgeting_urls)
    investment_splits = load_and_split(investment_urls)
    credit_splits = load_and_split(credit_urls)
    regulation_splits = load_and_split(regulation_urls)

    budgeting_vectorstore = FAISS.from_documents(
        documents=budgeting_splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    investment_vectorstore = FAISS.from_documents(
        documents=investment_splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    credit_vectorstore = FAISS.from_documents(
        documents=credit_splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    regulation_vectorstore = FAISS.from_documents(
        documents=regulation_splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Create retrievers
    budgeting_retriever = budgeting_vectorstore.as_retriever()
    investment_retriever = investment_vectorstore.as_retriever()
    credit_retriever = credit_vectorstore.as_retriever()
    regulation_retriever = regulation_vectorstore.as_retriever()

    return budgeting_retriever, investment_retriever, credit_retriever, regulation_retriever

# Set up retrievers for each agent
budgeting_retriever, investment_retriever, credit_retriever, regulation_retriever = setup_vector_stores()

# --- AGENTS ---

def budgeting_agent(state):
    print("---BUDGETING AGENT---")
    messages = state["messages"]
    question = messages[0].content
    docs = budgeting_retriever.invoke(question)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="Llama3-8b-8192")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def investment_agent(state):
    print("---INVESTMENT AGENT---")
    messages = state["messages"]
    question = messages[0].content
    docs = investment_retriever.invoke(question)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="Llama3-8b-8192")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def credit_agent(state):
    print("---CREDIT AGENT---")
    messages = state["messages"]
    question = messages[0].content
    docs = credit_retriever.invoke(question)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="Llama3-8b-8192")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def regulation_agent(state):
    print("---REGULATION AGENT---")
    messages = state["messages"]
    question = messages[0].content
    docs = regulation_retriever.invoke(question)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="Llama3-8b-8192")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def generic_agent(state):
    print("---GENERIC AGENT---")
    messages = state["messages"]
    question = messages[0].content
    generic_prompt = (
        "You are a helpful and honest AI assistant.\n"
        "Answer the following question as accurately and factually as possible.\n"
        "If you do not know the answer or do not have enough information, say: 'Sorry, I don't know.'\n"
        "Do not attempt to make up an answer or provide false information.\n"
        "Be concise and clear in your response.\n"
        "Keep your answer below 80 words.\n"
        "If the question is ambiguous, ask for clarification.\n"
        "If the question is outside your knowledge, admit it honestly.\n"
        "Always prioritize factual accuracy and honesty in your answers.\n"
        f"Question: {question}"
    )
    llm = ChatGroq(model="Llama3-8b-8192")
    response = llm.invoke([HumanMessage(content=generic_prompt)])
    if any(phrase in response.content.lower() for phrase in [
        "i don't know", "i am not sure", "cannot answer", "no information", "not enough information"
    ]):
        return {"messages": [HumanMessage(content="Sorry, I don't know.")]}
    else:
        return {"messages": [HumanMessage(content=response.content)]}

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_coordinator_agent(state):
    messages = state["messages"]
    question = messages[0].content.lower()

    # 1. Keyword Matching
    for agent, keywords in domain_keywords.items():
        if any(kw in question for kw in keywords):
            print(f"Hybrid Routing: Routed by keyword to {agent}")
            return {"route": agent, "messages": messages}

    # 2. Semantic Similarity
    question_embedding = embedding_model.embed_query(question)
    best_agent = None
    best_score = -float('inf')
    for agent, embeddings in domain_embeddings.items():
        score = max(
            cosine_similarity(question_embedding, emb)
            for emb in embeddings
        )
        if score > best_score:
            best_score = score
            best_agent = agent
    if best_score > 0.7:
        print(f"Hybrid Routing: Routed by semantic similarity to {best_agent} (score={best_score:.2f})")
        return {"route": best_agent, "messages": messages}

    # 3. LLM-based Routing Fallback
    print("Hybrid Routing: Using LLM fallback for routing")
    router_prompt = (
        "You are a router. Decide which agent should answer the following question:\n"
        "If it's about budgeting, answer 'budgeting_agent'.\n"
        "If it's about investment, answer 'investment_agent'.\n"
        "If it's about credit score, answer 'credit_agent'.\n"
        "If it's about regulation, answer 'regulation_agent'.\n"
        "If it's about something else, answer 'generic_agent'.\n"
        "Question: " + question
    )
    llm = ChatGroq(model="Llama3-8b-8192")
    response = llm.invoke([HumanMessage(content=router_prompt)])
    if "budgeting_agent" in response.content.lower():
        route = "budgeting_agent"
    elif "investment_agent" in response.content.lower():
        route = "investment_agent"
    elif "credit_agent" in response.content.lower():
        route = "credit_agent"
    elif "regulation_agent" in response.content.lower():
        route = "regulation_agent"
    else:
        route = "generic_agent"
    print(f"Hybrid Routing: Routed by LLM to {route}")
    return {"route": route, "messages": messages}

# --- WORKFLOW ---
def create_multiagent_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("coordinator", hybrid_coordinator_agent)
    workflow.add_node("budgeting_agent", budgeting_agent)
    workflow.add_node("investment_agent", investment_agent)
    workflow.add_node("credit_agent", credit_agent)
    workflow.add_node("regulation_agent", regulation_agent)
    workflow.add_node("generic_agent", generic_agent)
    workflow.add_edge(START, "coordinator")
    def route_decision(state):
        return state["route"]
    workflow.add_conditional_edges(
        "coordinator",
        route_decision,
        {
            "budgeting_agent": "budgeting_agent",
            "investment_agent": "investment_agent",
            "credit_agent": "credit_agent",
            "regulation_agent": "regulation_agent",
            "generic_agent": "generic_agent"
        }
    )
    workflow.add_edge("budgeting_agent", END)
    workflow.add_edge("investment_agent", END)
    workflow.add_edge("credit_agent", END)
    workflow.add_edge("regulation_agent", END)
    workflow.add_edge("generic_agent", END)
    return workflow.compile()

class FinancialEducationApp:
    def __init__(self):
        self.graph = create_multiagent_workflow()

    def run(self):
        st.markdown("<h4>Financial Agent [Multi-Agentic RAG]</h4>", unsafe_allow_html=True)
        st.write("Ask any question about budgeting, investment, credit or financial regulations.")

        user_query = st.text_input("Enter your financial question:")

        if st.button("Ask") and user_query.strip():
            with st.spinner("Thinking..."):
                try:
                    result = self.graph.invoke({"messages": [HumanMessage(content=user_query)]})
                    response = result['messages'][-1].content
                except Exception as e:
                    response = f"Error: {e}"
                st.markdown("**Response:**")
                st.write(response)
                
if __name__ == "__main__":
    app = FinancialEducationApp()
    app.run()

# def main():
#     graph = create_multiagent_workflow()
#     test_queries = [
#         "How do I start budgeting my monthly expenses?",
#         "What is a mutual fund and how does it work?",
#         "How can I improve my credit score?",
#         "What are my rights under financial regulations?",
#         "Tell me about the history of Bitcoin.",
#         "What is the best way to save for retirement?"
#     ]
#     print("Personalized Financial Education Multi-Agentic RAG System")
#     print("=" * 60)
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         print("-" * 30)
#         try:
#             result = graph.invoke({"messages": [HumanMessage(content=query)]})
#             print(f"Response: {result['messages'][-1].content}")
#         except Exception as e:
#             print(f"Error: {e}")
#         print()

# if __name__ == "__main__":
#     main()
