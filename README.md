# Multi-stage Reasoning with LangChain

A hands-on lab demonstrating multi-stage reasoning using LangChain, Azure Databricks, and Azure OpenAI.

## Overview

This project shows how to build a multi-chain AI system that combines vector search with language models to answer questions and generate content based on retrieved information.

## Prerequisites

- Azure subscription with administrative access
- Azure OpenAI service access approval

## Setup

### 1. Azure OpenAI Resource
- Create an Azure OpenAI resource
- Deploy models:
  - `gpt-4o` (10K tokens/min)
  - `text-embedding-ada-002` (10K tokens/min)
- Save your endpoint and API key

### 2. Azure Databricks Workspace
- Create a workspace in the same region as your OpenAI resource
- Create a single-node cluster (16.4 LTS ML runtime)
- Install required libraries:
  ```python
  %pip install langchain openai langchain_openai langchain-community faiss-cpu
  ```

## Main Components

### Vector Index & Embeddings
- Load sample documents
- Generate embeddings using Azure OpenAI
- Create FAISS vector index for similarity search

### Retriever Chain
- Build a retriever to search the vector index
- Create a QA system combining retriever + GPT-4o

### Multi-chain System
- Chain 1: Answer questions using retrieved context
- Chain 2: Generate social media posts from Chain 1 output
- Combine chains for complex multi-stage reasoning

## Cleanup

- Delete Azure OpenAI deployment/resource
- Terminate Databricks cluster
- Delete Databricks workspace (if no longer needed)

## Key Technologies

- **LangChain**: Framework for LLM applications
- **Azure OpenAI**: GPT-4o and embedding models
- **Azure Databricks**: Distributed processing platform
- **FAISS**: Vector similarity search

---
