import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create the Neo4jVector
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="moviePlots",
    node_label="Movie",
    text_node_property="plot",
    embedding_node_property="plotEmbedding"
)
# Create the retriever
retriever = neo4jvector.as_retriever()
# Create the prompt
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instructions),
        ("human", "{input}"),
    ]
)
# Create the chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever,
    question_answer_chain
)
# Create a function to call the chain
def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})