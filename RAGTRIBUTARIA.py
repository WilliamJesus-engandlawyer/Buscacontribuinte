# ============================================================
# Instalação das bibliotecas essenciais para o pipeline RAG
# ============================================================

!pip install -q lancedb sentence-transformers pdfplumber tqdm pyarrow
print("Tudo instalado! 🖤")  # 🖤 Wandinha observa em silêncio...
# ============================================================
# CÉLULA 2 — Upload dos PDFs e classificação automática
# ============================================================
from google.colab import files
import pdfplumber
from pathlib import Path
import json

print("Faça upload de TODOS os PDFs de uma vez (leis, decretos, CTM, CF, CTN, LAI, tudo junto)")
uploaded = files.upload()

pdfs = list(uploaded.keys())
print(f"\nCarregados {len(pdfs)} PDFs")

# ------------------------------------------------------------
# Função simples que classifica PDFs como Direito Formal/Material
# ------------------------------------------------------------
def classifica(pdf_nome, texto_inicio):
    texto = (pdf_nome + " " + texto_inicio).lower()
    formal_palavras = [
        "processo administrativo", "9.784", "lei 9784",
        "acesso à informação", "lai", "ética", "improbidade",
        "8.429", "transparência", "defesa", "recurso", "prazo recursal"
    ]
    if any(p in texto for p in formal_palavras):
        return "Direito Formal"
    return "Direito Material"  # default

categorias = {}
for pdf in pdfs:
    with pdfplumber.open(pdf) as p:
        inicio = "".join(page.extract_text() or "" for page in p.pages[:3])
    cat = classifica(Path(pdf).stem, inicio[:3000])
    categorias[pdf] = cat
    print(f"✓ {pdf} → {cat}")

# Salva resultado
with open("/content/categorias.json", "w", encoding="utf-8") as f:
    json.dump(categorias, f, ensure_ascii=False, indent=2)

print("\nClassificação salva em /content/categorias.json")


# ------------------------------------------------------------
# CÉLULA 3 — Processamento, limpeza, chunking e extração da norma
# ------------------------------------------------------------

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pdfplumber
import re
import json

# Modelo legal brasileiro de embeddings
model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

with open("/content/categorias.json") as f:
    categorias = json.load(f)

chunks = []
doc_id = 0

print("Processando PDFs e criando chunks... 🖤")  # 🖤 Como a Wandinha: sem emoções, apenas eficiência.

for pdf_path in tqdm(pdfs):
    with pdfplumber.open(pdf_path) as pdf:
        texto_completo = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # ------------------------------------------------------------
    # Tenta identificar automaticamente a norma (Lei XXXX/AAAA)
    # ------------------------------------------------------------
    inicio = texto_completo[:2000].upper()
    match = re.search(r"(LEI|DECRETO|LC)[\sªº.]*\s*N?º?\s*([\d.]+)\s*[/deDE]+\s*(\d{4})", inicio)
    norma = match.group(0) if match else Path(pdf_path).stem.replace("_", " ")

    categoria = categorias.get(pdf_path, "Direito Material")

    # ------------------------------------------------------------
    # CHUNKING com overlap
    # ------------------------------------------------------------
    passo = 800
    for i in range(0, len(texto_completo), passo):
        chunk = texto_completo[i:i+1200].strip()
        if len(chunk) > 150:
            chunks.append({
                "id": doc_id,
                "text": chunk,
                "source_file": pdf_path,
                "norma": norma,
                "categoria": categoria,
                "vigente": True,
                "hierarquia": 1 if "constituicao" in norma.lower() else 
                              2 if "ctn" in norma.lower() else 4
            })
            doc_id += 1

print(f"Gerados {len(chunks)} chunks com sucesso!")

# ============================================================
# CÉLULA 4 — Criação do banco LanceDB + embeddings
# ============================================================


import lancedb
import pyarrow as pa
from tqdm.auto import tqdm

db = lancedb.connect("./lancedb")

# ------------------------------------------------------------
# Criando os embeddings
# ------------------------------------------------------------
vectors = [model.encode(c["text"], normalize_embeddings=True).tolist()
           for c in tqdm(chunks, desc="Embeddings")]

# ------------------------------------------------------------
# Criação da tabela Arrow para armazenar no LanceDB
# ------------------------------------------------------------
table_data = pa.table({
    "id": [c["id"] for c in chunks],
    "text": [c["text"] for c in chunks],
    "source_file": [c["source_file"] for c in chunks],
    "norma": [c["norma"] for c in chunks],
    "categoria": [c["categoria"] for c in chunks],
    "vigente": [c["vigente"] for c in chunks],
    "hierarquia": [c["hierarquia"] for c in chunks],
    "vector": vectors
})

# Se existir tabela antiga, remove
if "laws" in db.table_names():
    db.drop_table("laws")

# Cria nova tabela vetorial
tbl = db.create_table("laws", data=table_data)

# Cria índice vetorial (cosine similarity)
tbl.create_index(metric="cosine", num_partitions=64)

print("LanceDB criado e indexado com sucesso! 🖤")
print(f"Total de chunks no banco: {len(chunks)}")
print("o mãozinha até que é competente programando... 🖤")  # 🖤 Como a Wandinha: sem emoções, apenas eficiência.
