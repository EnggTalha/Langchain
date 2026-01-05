# Generative AI using LangChain – Comprehensive Study Notes  
*(Rewritten & Enhanced from CampusX YouTube Series – Clean, Structured, and Ready for Use)*

A single, well-organized Markdown file perfect for:
- Self-study and revision
- Lecture notes or teaching material
- GitHub README / documentation
- Sharing with peers

---

## 1. Introduction to Generative AI

### What is Generative AI?
Generative AI (GenAI) is a category of artificial intelligence that can **create new content**, including:
- Text (answers, summaries, stories, code)
- Conversations
- Recommendations and decisions

The backbone of modern GenAI is **Large Language Models (LLMs)** such as GPT, Gemini, Llama, Claude, etc.

### Limitations of Using Raw LLM APIs
Direct API calls work for simple tasks, but real-world applications often require:
- Complex, dynamic prompts
- Multi-step reasoning
- Preservation of conversation context (memory)
- Grounding answers in private or custom documents
- Integration with external tools (search, databases, calculators)

**LangChain** is the framework that addresses these challenges elegantly.

---

## 2. What is LangChain?

**LangChain** is an open-source framework (Python & JavaScript) for building robust, scalable applications powered by LLMs.

### Key Features
- Modular components for prompts, models, chains, memory, etc.
- Easy integration with multiple LLM providers
- Built-in support for Retrieval-Augmented Generation (RAG)
- Agent capabilities (LLMs that can use tools)
- Active community and rich ecosystem

### Why Use LangChain?
- Cleaner, reusable code
- Faster prototyping and production deployment
- Handles complexity without spaghetti code

---

## 3. Core Components of LangChain

| Component              | Role                                                                 |
|------------------------|----------------------------------------------------------------------|
| **Models**             | LLMs, Chat Models, Embedding Models                                  |
| **Prompt Templates**   | Dynamic, reusable prompt structures                                  |
| **Chains**             | Pipelines combining prompts, models, and other steps                 |
| **Memory**             | Stores conversation history                                          |
| **Retrievers / Indexes**| Search over documents using embeddings                               |
| **Agents**             | LLMs that decide and use tools dynamically                            |

This document focuses on the foundational trio: **Models → Prompt Templates → Chains**.

---

## 4. Models in LangChain

### 4.1 LLMs (Text Completion Models)
For straightforward text generation.

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)  # Higher temperature = more creative
response = llm("Explain Generative AI in one sentence.")
print(response)