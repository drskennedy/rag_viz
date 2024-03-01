# Visualizing FAISS Vector Space Using Library spotlight

**Step-by-step guide on Medium**: [Visualizing FAISS Vector Space to Understand its Influence on RAG Performance](https://medium.com/ai-advances/visualizing-faiss-vector-space-to-understand-its-influence-on-rag-performance-14d71c6a4f47)
___
## Context
Retrieval-Augmented Generation (RAG) is a popular technique used to improve the text generation capability of an LLM by keeping it fact driven and reduce its hallucinations. RAG performance is directly influenced by the embeddings formed from the chosen documents.
In this project, we will develop a RAG application using FAISS vectorstore to find relevant documment snippets and use them as context by our LLM of choice, namely TinyLlama 1.1B Chat. For visualization, we used visualization library `renumics-spotlight` to visualize FAISS vecstorestore.
<br><br>
![System Design](/assets/architecture.png)
In addition, we looked at how the vector space property changes when varying certain vectorization parameters. Here is an example comparison:
<br><br>
![Visualizing Space with Changing Key Parameters](/assets/umap_comparison.png)
___
## How to Install
- Create and activate the environment:
```
$ python3.10 -m venv mychat
$ source mychat/bin/activate
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf from [TheBloke HF report](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) to directory `models`.
- Run script `main.py` to start the testing:
```
$ python main.py
```
___
## Quickstart
- To start the app, launch terminal from the project directory and run the following command:
```
$ source mychat/bin/activate
$ python main.py
```
- Here is a sample run:
```
$ python main.py
Q: What versions of TLS supported by Client Accelerator 6.3.0?
A: Client Accelerator 6.3.0 supports TLS versions 1.0 and 1.1 or 1.2. The supported TLS versions are listed in the table below:

| TLS Version | Supported |
|-------------|-----------|
| TLS 1.0 | Yes |
| TLS 1.1 | Yes |
| TLS 1.2 | Yes |

Note that TLS 1.0 and TLS 1.1 are no longer supported by some browsers and operating systems. Therefore, it's recommended to use TLS 1.2 for optimal performance and security.
```
Here is a screenshot of the visualization from this run:
<br><br>
![Vector Space Visualization](/assets/ui_screenshot.png)
___
## Key Libraries
- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Open-source library for efficient similarity search and clustering of dense vectors.
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Open-source pre-trained transformer model for embedding text to a dense vector space for tasks like cosine similarity calculation.
- **Spotlight: Visualization library to interactively explore unstructured datasets.

___
## Files and Content
- `models`: Directory hosting the downloaded LLM in GGUF format
- `opdf915_index`: Directory for FAISS index and vectorstore
- `main.py`: Main Python script to launch the application
- `LoadFVectorize.py`: Python script to load a pdf document, split and vectorize
- `requirements.txt`: List of Python dependencies (and version)
___

## References
- https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- https://github.com/Renumics/spotlight
