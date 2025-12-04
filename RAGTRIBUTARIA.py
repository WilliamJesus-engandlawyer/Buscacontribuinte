'''
certo esse código o que é? ele funcionária e funciona, muito bem como
o procedimento a ser seguido no google collab para criar uma rag. Ele funciona ótimo como um jypternotebbok
salvei aqui no git, para fins educacionais e documentação mesmo. Vamos a umas regrinhas tá?
NÃO EXECUTAR TUDO DE UMA VEZ!!
não de um copia e cola de tudo de uma vez, faça uma passo de cada vez. Separei por partes e dexei bonitinho o procedimento, faz cad um de  cada vez
'''

# ============================================================
# CÉLULA 1 — Instalação dos pacotes necessários
# Execute apenas se estiver em Google Colab
# ============================================================

!pip install -q sentence-transformers faiss-cpu pdfplumber pymupdf tqdm # o collab precisa desse ! não sei ao certo porque
# ============================================================
# CÉLULA 2 — Imports principais organizados
# ============================================================

from pathlib import Path
import json
import re
import numpy as np
from tqdm.auto import tqdm
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

# Upload e download no Google Colab
from google.colab import files
# ============================================================
# CÉLULA 3 — Upload dos PDFs + Classificação manual
# ============================================================

print("📤 Faça upload dos PDFs para o RAG:")
uploaded = files.upload()

PDF_FILES = []

print("\nClassifique cada arquivo:")

for file_name in uploaded.keys():

    print("\nArquivo:", file_name)
    print("Escolha o tipo:")
    print("1 - Constituição Federal")
    print("2 - Código Tributário Nacional (CTN)")
    print("3 - Lei Municipal")

    choice = input("Digite 1, 2 ou 3: ").strip()

    if choice == "1":
        meta = {
            "norma": "Constituição Federal",
            "tipo": "Constitucional",
            "hierarquia": 1
        }

    elif choice == "2":
        meta = {
            "norma": "Código Tributário Nacional",
            "tipo": "Lei Complementar Nacional",
            "hierarquia": 2
        }

    elif choice == "3":
        meta = {
            "norma": "Lei Municipal de Itaquaquecetuba",
            "tipo": "Lei Ordinária Municipal",
            "hierarquia": 3
        }

    else:
        raise ValueError("❌ Opção inválida.")

    PDF_FILES.append({
        "file": file_name,
        "meta": meta
    })

print("\n✅ PDFs classificados:")
for p in PDF_FILES:
    print("•", p["file"], "→", p["meta"]["norma"])

# Pasta de saída
OUTPUT_DIR = Path("/content/lei_rag_multi_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n📁 Saída configurada em:", OUTPUT_DIR)
# ============================================================
# CÉLULA 4 — Extração de texto dos PDFs (página por página)
# ============================================================

def extract_pdf_pages(pdf_files):
    pages = []

    for item in pdf_files:
        file = item["file"]
        meta = item["meta"]

        print("🔍 Extraindo:", file)

        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text()

                # Ignorar páginas vazias
                if not txt or not txt.strip():
                    continue

                pages.append({
                    "file": file,
                    "page": i + 1,
                    "norma": meta["norma"],
                    "tipo": meta["tipo"],
                    "hierarquia": meta["hierarquia"],
                    "text": txt
                })

    return pages


pages = extract_pdf_pages(PDF_FILES)

print("\n✅ Total de páginas extraídas:", len(pages))
# ============================================================
# CÉLULA 5 — Detecção de artigos jurídicos (“Art.”)
# ============================================================

ART_PATTERN = re.compile(r'(Art\.?\s+\d+[A-Za-z0-9\-]*[^\n]*)', flags=re.IGNORECASE)

def split_articles(pages):
    full_text = '\n\n'.join([f"[p{p['page']}]\n" + p['text'] for p in pages])
    full_text = re.sub(r'\r', '\n', full_text)

    articles = []
    matches = list(ART_PATTERN.finditer(full_text))

    if not matches:
        # fallback (não encontrou "Art.")
        print("⚠️ Nenhum artigo encontrado, usando blocos simples.")
        blocks = full_text.split("\n\n")
        return [{"id": i, "text": blk} for i, blk in enumerate(blocks)]

    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end].strip()
        if chunk:
            articles.append({"id": idx, "text": chunk})

    return articles


articles = split_articles(pages)

print("📄 Artigos detectados:", len(articles))
print("\n--- EXEMPLO ---\n")
print(articles[0]["text"][:800])
# ============================================================
# CÉLULA 6 — Dividir artigos grandes em chunk menores
# ============================================================

def chunk_article(text, max_chars=1200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= max_chars:
            current = (current + "\n" + p).strip() if current else p
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


documents = []
doc_id = 0

for art in articles:
    subs = chunk_article(art["text"])
    for s in subs:
        documents.append({
            "id": doc_id,
            "text": s
        })
        doc_id += 1

print("📦 Total de chunks gerados:", len(documents))
print("\n--- EXEMPLO DE CHUNK ---\n")
print(documents[0]["text"][:800])
# ============================================================
# CÉLULA 7 — Gerando embeddings com SentenceTransformers
# ============================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print("🔧 Carregando modelo:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

texts = [d["text"] for d in documents]
embs = model.encode(texts, show_progress_bar=True, batch_size=32).astype("float32")

# salvar
np.save(OUTPUT_DIR / "embeddings.npy", embs)
with open(OUTPUT_DIR / "documents.json", "w") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print("\n💾 Embeddings salvos em:", OUTPUT_DIR)
# ============================================================
# CÉLULA 8 — Criar o índice FAISS (similaridade por cosseno)
# ============================================================

faiss.normalize_L2(embs)
dim = embs.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embs)

faiss.write_index(index, str(OUTPUT_DIR / "lei_faiss.index"))

print("📚 Índice FAISS criado com", index.ntotal, "documentos")

# ============================================================
# CÉLULA 9 — Exportar tudo em .zip para baixar
# ============================================================

import shutil

zip_path = "/content/rag_tributaria_export.zip"

shutil.make_archive(
    base_name=zip_path.replace(".zip", ""),
    format="zip",
    root_dir=OUTPUT_DIR
)

print("📦 ZIP criado em:", zip_path)
files.download(zip_path)
# ---------



# ============================================================
# CÉLULA 10 — Função de busca (RAG) mais para testar se está funcionando 
# ============================================================

def retrieve(query, top_k=4):
    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, top_k)

    res = []
    for score, idx in zip(D[0], I[0]):
        res.append({
            "score": float(score),
            "id": int(idx),
            "text": documents[int(idx)]["text"]
        })

    return res


# TESTE
query = "O aposentado de Itaquaquecetuba com único imóvel possui isenção total de IPTU?"
results = retrieve(query)

for r in results:
    print("\n🔎 SCORE:", r["score"])
    print(r["text"][:800])
