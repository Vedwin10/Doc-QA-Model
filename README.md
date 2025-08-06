# Document Question Answering Model

A Document QA Model PoC based on SOTA research.

## Features
- Completely free to use and cost-efficient
- SOTA-based RAG pipeline
- Lightweight and and minimal GPU/CPU/TPU usage
- Fully created in Google Colab

## Components
- **Query Preprocessor:** Processes and cleans user queries to get rid of extra whitespace, normalize characters, and fix common spelling errors
- **PDF Parser:** Parses and loads PDF text into a local variable
- **Text Chunker:** Splits text into chunks with LangChain's RecursiveTextSplitter
- **Chunk Embedder:** Generates vector embeddings from text chunks using SentenceTransformers
- **Vector Store:** Uses Facebook AI Similarity Search (FAISS) to store vector embeddings and retrieve relevant embeddings
- **Reranker:** Reranks the relevance of reach chunk based on its relation to the query using a CrossEncoders
- **Prompt Engineering:** Deepseek-R1-specific prompt engineering for accurate LLM responses and reduction of model hallucination
- **LLM:** Deepseek R1 API calls with relevant chunks and processed query inserted into an engineered prompt

## Teck Stack
**Languages:** Python
**Libraries:** PyTorch, LangChain, PyMuPDF, SentenceTransformers, NumPy, FAISS, and more

## Implementation Steps
1. Sign up for OpenRouter's deepseek/deepseek-r1:free model (up to 50 requests/day), and plug in your API key in the line:
```python
my_api_key=""
```
2. Import `Doc-QA-Model.ipynb` into Google Colab
3. Import all other files in this codebase into Google Colab's file system (under ../content/)
4. Select 'GPU' as the runtime type
5. Run each cell individually from the beginning (note: the kernal will restart for condocolab, just continue from where you left off)

## Demo


## 3-Page Research Paper (By Vedik Upadhyay):
A more in-depth paper on my solution with publication references is detailed in `DocQA Final Paper.pdf`

## Future Improvements
- Add a user interface
- Add user-based authentication
- Secure API calls
- Make this deployable
- Organize file structure