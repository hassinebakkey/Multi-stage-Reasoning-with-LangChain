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
print("Chargement de la configuration...")
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")

if not endpoint or not key:
    raise ValueError("Erreur: AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_KEY doivent être définis dans .env")

print(f"Endpoint configuré: {endpoint}")

# ========================================
# CRÉATION DES DOCUMENTS (Sample Dataset)
# ========================================
print("\nCréation de la base de connaissances...")

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

print(f"{len(documents)} documents créés")

# ========================================
# GÉNÉRATION DES EMBEDDINGS
# ========================================
print("\nInitialisation du modèle d'embeddings...")

embedding_function = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    azure_endpoint=endpoint,
    openai_api_key=key,
    chunk_size=1
)

print("Modèle d'embeddings initialisé")

# ========================================
# CRÉATION DE L'INDEX VECTORIEL (FAISS)
# ========================================
print("\nCréation de l'index vectoriel...")

# Créer l'index FAISS
index = faiss.IndexFlatL2(
    len(embedding_function.embed_query("Azure Databricks is a fast, easy, and collaborative Apache Spark-based analytics platform."))
)

# Créer le vector store
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Ajouter les documents au vector store
vector_store.add_documents(documents=documents, ids=ids)
print("Documents vectorisés et indexés")

# ========================================
# CRÉATION DU RETRIEVER
# ========================================
print("\nCréation du retriever...")

retriever = VectorStoreRetriever(vectorstore=vector_store)
print("Retriever prêt")

# ========================================
# INITIALISATION DU LLM (GPT-4o)
# ========================================
print("\nInitialisation du modèle GPT-4o...")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    azure_endpoint=endpoint,
    api_version="2023-03-15-preview",
    openai_api_key=key,
)

print("GPT-4o initialisé")

# ========================================
# CRÉATION DE LA CHAÎNE 1 : QA SYSTEM
# ========================================
print("\nCréation de la chaîne QA...")

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

# Chaîne QA avec LCEL
qa_chain1 = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt1
    | llm
    | StrOutputParser()
)

print("Chaîne QA créée")

# ========================================
# CRÉATION DE LA CHAÎNE 2 : SOCIAL MEDIA
# ========================================
print("\nCréation de la chaîne multi-stage...")

prompt2 = ChatPromptTemplate.from_template(
    "Create a social media post based on this summary: {summary}"
)

qa_chain2 = ({"summary": qa_chain1} | prompt2 | llm | StrOutputParser())

print("Chaîne multi-stage créée")

# ========================================
# TESTS
# ========================================
print("\n" + "="*60)
print("TEST 1 : Question directe (Chaîne QA)")
print("="*60)

question1 = "What is Azure Databricks?"
print(f"\nQuestion: {question1}")
print("\nRéponse:")

result1 = qa_chain1.invoke(question1)
print(result1)

print("\n" + "="*60)
print("TEST 2 : Multi-stage reasoning (QA → Social Media)")
print("="*60)

question2 = "How can we use LangChain?"
print(f"\nQuestion: {question2}")
print("\nPost sur les réseaux sociaux:")

result2 = qa_chain2.invoke(question2)
print(result2)

print("\n" + "="*60)
print("TEST 3 : Question sur GPT-4")
print("="*60)

question3 = "What is GPT-4?"
print(f"\nQuestion: {question3}")
print("\nPost sur les réseaux sociaux:")

result3 = qa_chain2.invoke(question3)
print(result3)

print("\n" + "="*60)
print("Tous les tests terminés avec succès!")
print("="*60)
