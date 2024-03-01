# LoadFVectorize.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# access some online pdf, load, vectorize and commit to disk
def load_doc() -> List[Document]:
    loader = OnlinePDFLoader("https://support.riverbed.com/bin/support/download?did=7q6behe7hotvnpqd9a03h1dji&version=9.15.0")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# vectorize, commit to disk and create a BM25 retriever
def vectorize(embeddings_model) -> FAISS:
    docs = load_doc()
    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local("./opdf915_index")
    return db

# attempts to load vectorstore from disk
def load_db() -> FAISS:
    embeddings_model = HuggingFaceEmbeddings()
    try:
        db = FAISS.load_local("./opdf915_index", embeddings_model)
    except Exception as e:
        print(f'Exception: {e}\nNo index on disk, creating new...')
        db = vectorize(embeddings_model)
    return db
