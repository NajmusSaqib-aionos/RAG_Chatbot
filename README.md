# RAG-Based Study & Learning Chatbot

An intelligent chatbot leveraging Retrieval-Augmented Generation (RAG) to assist users in studying and learning from unstructured documents such as PDFs, textbooks, and articles.

---

## Objective

The chatbot enables conversational interaction to help users:
- Ask natural language questions about their study material  
- Summarize chapters and documents  
- Quiz themselves for better knowledge retention  
- Access relevant, context-aware answers using advanced retrieval and generation techniques

---

## Architecture & Workflow Overview

The system follows a multi-phase pipeline that ensures accurate and efficient responses:

1. **Initial Setup Phase**  
   - User uploads study documents (PDFs, text) through the Chat UI  
   - The Query Processor parses and splits documents into manageable chunks  
   - Transformer-based embeddings are generated and stored in a vector database (Chroma)  
   - Once indexing completes, the system is ready to accept queries

2. **Query Processing Phase**  
   - The Chat UI captures user questions  
   - The Query Processor analyzes the intent and converts queries into embeddings for retrieval

3. **Retrieval Phase**  
   - The Retriever searches the vector database for the top-k most similar document chunks  
   - Retrieved chunks form the candidate context for the answer

4. **Re-ranking Phase**  
   - Candidate chunks are scored and reordered based on contextual relevance to the query  
   - The best-ranked content is selected for answer generation

5. **Generation Phase**  
   - A Large Language Model (Groq LLM) receives the user query along with the relevant context  
   - The model generates a human-readable, detailed answer combining retrieved knowledge and general domain knowledge

6. **Response and Logging Phase**  
   - The generated response is shown to the user with citations  
   - Interaction data (query, context, response, metadata) is logged for analytics and audit

---

## Features

- Multi-document support with PDF ingestion  
- Semantic search over document embeddings  
- Conversational Q&A with document context  
- Chapter and document summarization capabilities  
- Persistent vector database for efficient retrieval  
- User-friendly Streamlit-based chat interface
