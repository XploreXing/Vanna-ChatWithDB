import os
if 'MILVUS_URI' in os.environ:
    del os.environ['MILVUS_URI']

import streamlit as st
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient,model
from vanna.milvus import Milvus_VectorStore
from vanna.openai import OpenAI_Chat



class VannaMilvus(Milvus_VectorStore,OpenAI_Chat):
    def __init__(self, llm_client, config=None):
        Milvus_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=llm_client,config=config)

class EmbeddingWrapper:
    def __init__(self, embedder):
        self.embedder = embedder
    def encode_documents(self, texts):
        result=self.embedder.embed_documents(texts)
        return [np.array(r) for r in result]
    
    def encode_queries(self,texts):
        embeddings=[]
        for text in texts:
            embeddings.append(self.embedder.embed_query(text))
        return embeddings


@st.cache_resource(ttl=3600)
def setup_vanna():
    llm_client=OpenAI(
        base_url=st.secrets["LLM_BASE_URL"],
        api_key=st.secrets["NVIDIA_API_KEY"],
    )
    silicon_embedder=OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url=st.secrets["SILICONFLOW_BASE_URL"],
        api_key=st.secrets["SILICONFLOW_API_KEY"],
    )
    vanna_embedder=EmbeddingWrapper(silicon_embedder)
    milvus_client=MilvusClient(st.secrets["MILVUS_URI"])
    config={
        "model": st.secrets["LLM_MODEL"],
        "milvus_client": milvus_client,
        "embedding_function:":vanna_embedder,
        "n_results":5
    }
    vanna=VannaMilvus(llm_client,config=config)
    vanna.connect_to_sqlite(st.secrets["SQLITE_PATH"])
    print("Vanna connected to SQLite successfully")
    return vanna

@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question:str):
    vanna=setup_vanna()
    sql=vanna.generate_sql(question)
    return sql

@st.cache_data(show_spinner="Checking if SQL query is valid ...")
def is_sql_valid_cached(sql:str):
    vanna=setup_vanna()
    return vanna.is_sql_valid(sql)

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()
    
@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql:str):
    vanna=setup_vanna()
    return vanna.run_sql(sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code

@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)

@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)