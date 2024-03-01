# main.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
import LoadFVectorize
from renumics import spotlight
import pandas as pd
import numpy as np

def visualize_distance(db,question,answer) -> None:
    embeddings_model = HuggingFaceEmbeddings()
    vs = db.__dict__.get("docstore")
    index_list = db.__dict__.get("index_to_docstore_id").values()
    doc_cnt = db.index.ntotal
    doc_list = list()
    embeddings_vec = db.index.reconstruct_n()
    # create a list of dicts
    for i,id in enumerate(index_list):
        a_doc = vs.search(id)
        doc_list.append([id,a_doc.metadata.get("source"),a_doc.page_content,embeddings_vec[i]])

    # create a dataframe 
    df = pd.DataFrame(doc_list,columns=['id','metadata','document','embedding'])

    # add rows for question and answer
    question_df = pd.DataFrame(
        {
            "id": "question",
            "question": question,
            "embedding": [embeddings_model.embed_query(question)],
        })
    answer_df = pd.DataFrame(
        {
            "id": "answer",
            "answer": answer,
            "embedding": [embeddings_model.embed_query(answer)],
        })
    df = pd.concat([question_df, answer_df, df])

    question_embedding = embeddings_model.embed_query(question)
    # add column for vector distance
    df["dist"] = df.apply(
        lambda row: np.linalg.norm(
            np.array(row["embedding"]) - question_embedding
        ), axis=1,)
    spotlight.show(df)                                                                                                                    

# Prompt template 
qa_template = """<|system|>
You are a friendly chatbot who always responds in a precise manner. If answer is 
unknown to you, you will politely say so.
Use the following context to answer the question below:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""

# create a prompt instance 
QA_PROMPT = PromptTemplate.from_template(qa_template)

# download model beforehand
llm = LlamaCpp(
    model_path="./models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    temperature=0.01,
    max_tokens=2000,
    top_p=1,
    verbose=False,
    n_ctx=2048
)
db = LoadFVectorize.load_db()
faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}, max_tokens_limit=2000)

# custom QA Chain 
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=faiss_retriever,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# question
query = 'What versions of TLS supported by Client Accelerator 6.3.0?'
result = qa_chain({"query": query})
print(f'--------------\nQ: {query}\nA: {result["result"]}')

# visualise
visualize_distance(db,query,result["result"])

