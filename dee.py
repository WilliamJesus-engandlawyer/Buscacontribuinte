# ============================================================
# app.py — RAG TRIBUTÁRIO ITAQUAQUECETUBA NO STREAMLIT 
# Rode com: streamlit run app.py (o nome do arquivo py)
# ============================================================




# 1. Cria ambiente virtual (só na primeira vez)
#python -m venv venv

# 2. Ativa
#venv\Scripts\activate

# 3. Instala as dependências
#pip install streamlit lancedb sentence-transformers ollama tqdm

# 4. RODA O STREAMLIT
#streamlit run app.py


# ---------------------
import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import ollama
from datetime import datetime
import os

# ---------------------- CONFIGURAÇÃO DA PÁGINA ----------------------
st.set_page_config(
    page_title="Dr. Rafael Torres – Advogado Tributário Virtual",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Dr. Gabriel")
st.markdown("**Procurador municipal • OAB/SP • 1000 anos de experiência**  \nConsulta gratuita 24h sobre IPTU, ISS, ITBI e leis municipais de Itaquaquecetuba")

# ---------------------- CARREGA TUDO (só na primeira vez) ----------------------
@st.cache_resource
def load_rag():
    st.info("Carregando base de leis e inteligência artificial... (só acontece uma vez)")
    
    # 1. LanceDB
    db = lancedb.connect("./lancedb")
    tbl = db.open_table("laws")
    
    # 2. Modelo de embedding (mesmo do Colab)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    return tbl, model

tbl, dense_model = load_rag()

# ---------------------- FUNÇÃO DE BUSCA (igual Célula 7 Turbo) ----------------------
def busca_rag_streamlit(pergunta, top_k=6):
    query_vec = dense_model.encode(pergunta, normalize_embeddings=True).astype("float32")
    
    where_clauses = ["vigente = true", "hierarquia <= 3"]
    keyword_boost = []
    p_lower = pergunta.lower()
    
    if any(x in p_lower for x in ["aposentado", "pensionista", "idoso"]):
        keyword_boost.append("text LIKE '%aposentado%' OR text LIKE '%pensionista%' OR text LIKE '%idoso%'")
    if "isenção" in p_lower or "imunidade" in p_lower:
        keyword_boost.append("text LIKE '%isenção%' OR text LIKE '%imunidade%'")
    if "alíquota" in p_lower:
        keyword_boost.append("text LIKE '%alíquota%'")
    if "parcelamento" in p_lower:
        keyword_boost.append("text LIKE '%parcelamento%'")
    
    search = tbl.search(query_vec).metric("cosine").limit(top_k*5)
    search = search.where(" AND ".join(where_clauses))
    if keyword_boost:
        search = search.where(" OR ".join(keyword_boost), prefilter=True)
    
    return search.to_list()[:top_k]

# ---------------------- CHAT COM O ADVOGADO ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Em que posso ajudá-lo hoje com relação a tributos municipais de Itaquaquecetuba?"}
    ]

# Exibe histórico
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Input do usuário
if prompt := st.chat_input("Ex: Sou aposentado, tenho isenção de IPTU?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando a legislação municipal..."):
            # 1. Busca no RAG
            docs = busca_rag_streamlit(prompt, top_k=6)
            contexto = "\n\n".join([
                f"FONTE: {d['norma']} {d.get('numero','')}/{d.get('ano','')} ─ {d.get('source_file','')}\n{d['text'][:1000]}"
                for d in docs
            ])
            
            # 2. Prompt pro Ollama
            mensagem = f"""Você é o Dr. Rafael Torres, advogado tributarista.
Responda de forma profissional, educada e completa à pergunta abaixo, usando APENAS as leis fornecidas.

PERGUNTA: {prompt}
LEIS: {contexto}

Comece com "Prezado(a) Cliente," e termine oferecendo mais ajuda."""
            
            resposta = ollama.chat(model="gemma2:9b-instruct-qat", messages=[{"role": "user", "content": mensagem}])
            texto = resposta['message']['content']
            
            st.write(texto)
            st.session_state.messages.append({"role": "assistant", "content": texto})

# ---------------------- SIDEBAR COM INFORMAÇÕES ----------------------
with st.sidebar:
    st.header("ℹ️ Sobre este assistente")
    st.write("""
    - Baseado em **todas as leis municipais vigentes** de Itaquaquecetuba
    - Usa inteligência artificial (Gemma2 9B + RAG)
    - 100% offline e gratuito
    - Criado em Dezembro/2025
    """)
    st.divider()
    st.caption("Dica: pergunte qualquer coisa sobre IPTU, ISS, ITBI, parcelamento, isenção de aposentado, imunidade de templo...")