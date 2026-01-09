import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

# ========================================
# CONFIGURATION
# ========================================
print("Loading configuration...")
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")

if not endpoint or not key:
    raise ValueError("Error: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be defined in .env")

print(f"Endpoint configured: {endpoint}")

# ========================================
# CREATING DOCUMENTS (Sample Dataset)
# ========================================
print("\nCreating knowledge base...")

documents = [
    Document(
        page_content="Azure Databricks is a fast, easy, and collaborative Apache Spark-based analytics platform.",
        metadata={"date_created": "2024-08-22"}
    ),
    Document(
        page_content="LangChain is a framework designed to simplify the creation of applications using large language models.",
        metadata={"date_created": "2024-08-22"}
    ),
    Document(
        page_content="GPT-4 is a powerful language model developed by OpenAI.",
        metadata={"date_created": "2024-08-22"}
    )
]
ids = ["1", "2", "3"]

print(f"{len(documents)} documents created")

# ========================================
# GENERATING EMBEDDINGS
# ========================================
print("\nInitializing embedding model...")

embedding_function = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    azure_endpoint=endpoint,
    openai_api_key=key,
    chunk_size=1
)

print("Embedding model initialized")

# ========================================
# CREATING VECTOR INDEX (FAISS)
# ========================================
print("\nCreating vector index...")

# Create FAISS index
index = faiss.IndexFlatL2(
    len(embedding_function.embed_query("Azure Databricks is a fast, easy, and collaborative Apache Spark-based analytics platform."))
)

# Create vector store
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add documents to vector store
vector_store.add_documents(documents=documents, ids=ids)
print("Documents vectorized and indexed")

# ========================================
# CREATING RETRIEVER
# ========================================
print("\nCreating retriever...")

retriever = VectorStoreRetriever(vectorstore=vector_store)
print("Retriever ready")

# ========================================
# INITIALIZING LLM (GPT-4o)
# ========================================
print("\nInitializing GPT-4o model...")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    azure_endpoint=endpoint,
    api_version="2023-03-15-preview",
    openai_api_key=key,
)

print("GPT-4o initialized")

# ========================================
# CREATING CHAIN 1: QA SYSTEM
# ========================================
print("\nCreating QA chain...")

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)

prompt1 = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# QA Chain with LCEL
qa_chain1 = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt1
    | llm
    | StrOutputParser()
)

print("QA chain created")

# ========================================
# CREATING CHAIN 2: SOCIAL MEDIA
# ========================================
print("\nCreating multi-stage chain...")

prompt2 = ChatPromptTemplate.from_template(
    "Create a social media post based on this summary: {summary}"
)

qa_chain2 = ({"summary": qa_chain1} | prompt2 | llm | StrOutputParser())

print("Multi-stage chain created")

# ========================================
# TESTS
# ========================================
print("\n" + "="*60)
print("TEST 1: Direct Question (QA Chain)")
print("="*60)

question1 = "What is Azure Databricks?"
print(f"\nQuestion: {question1}")
print("\nAnswer:")

result1 = qa_chain1.invoke(question1)
print(result1)

print("\n" + "="*60)
print("TEST 2: Multi-stage Reasoning (QA → Social Media)")
print("="*60)

question2 = "How can we use LangChain?"
print(f"\nQuestion: {question2}")
print("\nSocial media post:")

result2 = qa_chain2.invoke(question2)
print(result2)

print("\n" + "="*60)
print("TEST 3: Question about GPT-4")
print("="*60)

question3 = "What is GPT-4?"
print(f"\nQuestion: {question3}")
print("\nSocial media post:")

result3 = qa_chain2.invoke(question3)
print(result3)

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)