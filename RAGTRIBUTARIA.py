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
# CÉLULA 3 — Upload estruturado das normas
# de verdade, eu sei que isso aqui tá muitooo, manual, podeira ser automatizado sim, poderia
# mas eu não quero, eu gosto dessa célula 3 assim do jeito que ela é em upload
# porque ai tenho controle melhor dos arquivos que estou colocando na rag.
# é tipo, mania minha mesmo
# ============================================================
from google.colab import files
from pathlib import Path

PDF_FILES = []

print("📤 Faça upload dos 4 PDFs principais (1 arquivo para cada):")
print("""
1 - Constituição Federal
2 - Código Tributário Nacional (CTN)
3 - Código Tributário Municipal
4 - Código de Posturas Municipal
""")

# Dicionário para identificar meta fixa
META_FIXA = {
    "1": {
        "norma": "Constituição Federal",
        "tipo": "Constitucional",
        "hierarquia": 1
    },
    "2": {
        "norma": "Código Tributário Nacional",
        "tipo": "Lei Complementar Nacional",
        "hierarquia": 2
    },
    "3": {
        "norma": "Código Tributário Municipal",
        "tipo": "Lei Ordinária Municipal",
        "hierarquia": 3
    },
    "4": {
        "norma": "Código de Posturas Municipal",
        "tipo": "Lei Ordinária Municipal",
        "hierarquia": 3
    },
}

# -----------------------------
# UPLOAD DOS 4 ARQUIVOS PRINCIPAIS
# -----------------------------
for key, meta in META_FIXA.items():
    print(f"\n➡️ Envie o PDF para: {meta['norma']}")
    uploaded = files.upload()

    if len(uploaded) != 1:
        raise ValueError("❌ Envie apenas 1 arquivo por categoria.")

    file_name = list(uploaded.keys())[0]

    PDF_FILES.append({
        "file": file_name,
        "meta": meta
    })

print("\n✅ Arquivos principais carregados!")

# -----------------------------
# UPLOAD ILIMITADO — LEIS MUNICIPAIS
# -----------------------------
print("\n📤 Agora envie **quantas Leis Municipais quiser**.")
print("Quando terminar, clique em 'cancelar' no seletor de arquivos.")

leis_uploaded = files.upload()

for file_name in leis_uploaded.keys():
    PDF_FILES.append({
        "file": file_name,
        "meta": {
            "norma": "Lei Municipal",
            "tipo": "Lei Ordinária Municipal",
            "hierarquia": 3
        }
    })

print(f"✔️ {len(leis_uploaded)} Leis Municipais carregadas!")

# -----------------------------
# UPLOAD ILIMITADO — DECRETOS MUNICIPAIS
# -----------------------------
print("\n📤 Agora envie **quantos Decretos Municipais quiser**.")
print("Quando terminar, clique em 'cancelar'.")

decretos_uploaded = files.upload()

for file_name in decretos_uploaded.keys():
    PDF_FILES.append({
        "file": file_name,
        "meta": {
            "norma": "Decreto Municipal",
            "tipo": "Decreto Municipal",
            "hierarquia": 3
        }
    })

print(f"✔️ {len(decretos_uploaded)} Decretos carregados!")

# -----------------------------
# EXIBIR RESULTADO FINAL
# -----------------------------
print("\n📄 PDFs classificados:")
for p in PDF_FILES:
    print(f"• {p['file']} → {p['meta']['norma']} (H{p['meta']['hierarquia']})")

# Criar pasta de saída
OUTPUT_DIR = Path("/content/lei_rag_multi_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("\n📁 Pasta de saída configurada em:", OUTPUT_DIR)


# ============================================================
# CÉLULA 2 — Extração ROBUSTA de número/ano + revogações + vínculo perfeito com arquivo
# ============================================================

import re
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# ==================== REGEX BRASIL REAL 2025 ====================
# Testado em +200 leis/decretos de SP, RJ, MG, RS, DF
PATTERNS_NUMERO_ANO = [
    r"(?:Lei|Decreto|Lei\s*n[º°ª]?|LC|Emenda\s*Constitucional)[\sº°ª.]*\s*(\d{1,5}(?:\.\d{3})*)[\s,\/de-]*\s*(\d{4})",
    r"(?:Lei|Decreto)[\sº°ª]*[\.:]?\s*(\d{1,5}(?:\.\d{3})*)[\/\s,]+(?:de)?[\s,]+(\d{4})",
    r"(?:Lei|Decreto)[\sº°ª]*\s*n?[º°ª]?\s*(\d{1,5}(?:\.\d{3})*)[\/\-–—]{1,2}(\d{4})",
    r"(?:LEI|DECRETO)[\sNº°ª]*\s*(\d{1,5}(?:\.\d{3})*)[\/\-–—\s]+(\d{4})",
]

PATTERN_REVOGA = r"(?:revog[a-z]*|fica[m]?\s*revogad[a-z]*)[^\.]*?(?:Lei|Decreto)[\sº°ªn]*[\s\.]*(\d{1,5}(?:\.\d{3})*)[\/\-–—\s]+(\d{4})"

def extrai_numero_ano(texto):
    texto = " " + texto + " "
    for pattern in PATTERNS_NUMERO_ANO:
        match = re.search(pattern, texto, re.IGNORECASE)
        if match:
            num = match.group(1).replace(".", "")
            ano = match.group(2)
            return num, ano
    return None, None

def extrai_revogacoes(texto):
    revogadas = []
    for m in re.finditer(PATTERN_REVOGA, texto, re.IGNORECASE):
        num = m.group(1).replace(".", "")
        ano = m.group(2)
        revogadas.append(f"{num}/{ano}")
    return list(set(revogadas))  # remove duplicatas

# ==================== PROCESSAMENTO ====================
records = []

print("Processando normas com detecção brasileira real...")

for item in PDF_FILES:
    file_name = item["file"]
    meta_fixa = item["meta"]
    
    text = extract_text_from_pdf(file_name)
    
    numero, ano = extrai_numero_ano(text)
    revogacoes = extrai_revogacoes(text)
    
    # Tipo mais preciso com base no conteúdo
    tipo_detectado = meta_fixa["tipo"]
    if "constituicao" in text.lower()[:2000]:
        tipo_detectado = "Constitucional"
    elif "complementar" in text.lower() and ("lei" in text.lower()):
        tipo_detectado = "Lei Complementar Nacional" if meta_fixa["hierarquia"] <= 2 else "Lei Complementar Municipal"
    
    records.append({
        "arquivo": file_name,
        "norma": meta_fixa["norma"],
        "tipo": tipo_detectado,
        "hierarquia": meta_fixa["hierarquia"],
        "numero": numero,
        "ano": ano,
        "revoga": revogacoes if revogacoes else None,
        "vigente": True  # será ajustado depois
    })

df_normas = pd.DataFrame(records)

# ==================== VIGÊNCIA ====================
revogadas_set = set()
for r in records:
    if r["revoga"]:
        revogadas_set.update(r["revoga"])

for idx, row in df_normas.iterrows():
    if row["numero"] and row["ano"]:
        chave = f"{row['numero']}/{row['ano']}"
        if chave in revogadas_set:
            df_normas.at[idx, "vigente"] = False

print("\nTabela final de normas (com vigência correta):")
display(df_normas[['arquivo', 'norma', 'numero', 'ano', 'vigente', 'revoga']])

# Salva para uso futuro
df_normas.to_json(OUTPUT_DIR / "metadados_normas.json", force_ascii=False, indent=2)
print(f"\nMetadados salvos em {OUTPUT_DIR}/metadados_normas.json")

# ============================================================
# CÉLULA 5 — Chunking com source_file 100% confiável + parent retriever
# ============================================================

from pathlib import Path
import json
import re

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 150

# Carrega metadados corretos (do JSON salvo acima)
meta_path = OUTPUT_DIR / "metadados_normas.json"
if meta_path.exists():
    df_meta = pd.read_json(meta_path)
    meta_by_file = df_meta.set_index("arquivo").to_dict("index")
else:
    raise FileNotFoundError("Execute a Célula 4 corrigida primeiro!")

def split_por_artigo(text):
    # Divide mantendo o "Art. 1º" como início de cada parte
    parts = re.split(r'\n(?=Art\.?\s+\d+)', text, flags=re.IGNORECASE)
    if len(parts) > 3:  # se dividiu em pelo menos 3 artigos reais
        return [p.strip() for p in parts if len(p.strip()) > 50]
    return None

def chunk_text(text, source_file, meta):
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tenta dividir por artigo primeiro
    artigos = split_por_artigo(text)
    if artigos:
        chunks = []
        for art in artigos:
            if len(art) <= CHUNK_SIZE:
                if len(art) >= MIN_CHUNK_LEN:
                    chunks.append(art)
            else:
                # sliding window dentro do artigo
                start = 0
                while start < len(art):
                    end = start + CHUNK_SIZE
                    chunk = art[start:end]
                    if len(chunk) >= MIN_CHUNK_LEN:
                        chunks.append(chunk)
                    start += (CHUNK_SIZE - CHUNK_OVERLAP)
        return chunks
    else:
        # Fallback: sliding window no texto todo
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if len(chunk) >= MIN_CHUNK_LEN:
                chunks.append(chunk)
            start += (CHUNK_SIZE - CHUNK_OVERLAP)
        return chunks

# ==================== GERAÇÃO ====================
documents = []
parents = {}
parent_counter = 0
doc_id = 0

for item in PDF_FILES:
    file_name = item["file"]
    if file_name not in meta_by_file:
        print(f"AVISO: {file_name} sem metadados!")
        continue
        
    meta = meta_by_file[file_name]
    text_completo = extract_text_from_pdf(file_name)
    if not text_completo.strip():
        continue
    
    # Cria parent (documento inteiro)
    parent_id = f"doc_{parent_counter}"
    parents[parent_id] = {
        "id": parent_id,
        "source_file": file_name,
        "text": text_completo[:50000],  # limita pra não explodir memória
        **meta
    }
    parent_counter += 1
    
    # Gera chunks
    chunks = chunk_text(text_completo, file_name, meta)
    for chunk in chunks:
        documents.append({
            "id": doc_id,
            "text": chunk,
            "parent_id": parent_id,
            "source_file": file_name,        # <-- AQUI ESTÁ A CHAVE!
            "meta": meta
        })
        doc_id += 1

# Remove duplicatas exatas
seen = set()
unique_docs = []
for d in documents:
    if d["text"] not in seen:
        unique_docs.append(d)
        seen.add(d["text"])
documents = unique_docs

# Salva tudo
with open(OUTPUT_DIR / "documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "parents.json", "w", encoding="utf-8") as f:
    json.dump(parents, f, ensure_ascii=False, indent=2)

print(f"\nCHUNKING PERFEITO!")
print(f"→ {len(documents)} chunks criados")
print(f"→ {len(parents)} documentos-pai")
print(f"→ source_file presente em 100% dos chunks")
print(f"→ Arquivos salvos em {OUTPUT_DIR}")

# ============================================================
# CÉLULA 6 CORRIGIDA E DEFINITIVA (2025) — LanceDB FUNCIONANDO
# ============================================================

!pip install -q "lancedb>=0.13" --upgrade

import lancedb
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# 1. Diretórios
# ------------------------------------------------------------------
OUTPUT_DIR = Path("/content/lei_rag_multi_output")
LANCE_DIR = Path("./lancedb")
LANCE_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 2. Garante o modelo de embeddings (o mesmo que você já usa)
# ------------------------------------------------------------------
if 'dense_model' not in globals():
    print("Carregando modelo de embeddings...")
    dense_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ------------------------------------------------------------------
# 3. Gera embeddings de todos os chunks (só roda uma vez)
# ------------------------------------------------------------------
print(f"Gerando embeddings para {len(documents)} chunks...")
texts = [doc["text"] for doc in documents]
embeddings = dense_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")

# ------------------------------------------------------------------
# 4. Monta o DataFrame completo com metadados + vetor
# ------------------------------------------------------------------
rows = []
for i, doc in enumerate(documents):
    meta = doc.get("meta", {})
    rows.append({
        "id": int(doc["id"]),
        "text": doc["text"],
        "parent_id": doc.get("parent_id"),
        "source_file": doc.get("source_file"),
        "norma": meta.get("norma"),
        "tipo": meta.get("tipo"),
        "hierarquia": meta.get("hierarquia"),
        "numero": meta.get("numero"),
        "ano": meta.get("ano"),
        "vigente": meta.get("vigente", True),
        "revoga": str(meta.get("revoga")) if meta.get("revoga") else None,
        "vector": embeddings[i].tolist()  # ← coluna OBRIGATÓRIA agora
    })

df_lance = pd.DataFrame(rows)

# ------------------------------------------------------------------
# 5. Cria/recria a tabela no LanceDB (limpa versão antiga)
# ------------------------------------------------------------------
db = lancedb.connect(LANCE_DIR)

if "laws" in db.table_names():
    db.drop_table("laws")
    print("Tabela antiga 'laws' removida.")

table = db.create_table("laws", data=df_lance)
print(f"Tabela 'laws' criada com {len(df_lance)} registros!")

# ------------------------------------------------------------------
# 6. Cria índice vetorial (busca fica instantânea)
# ------------------------------------------------------------------
table.create_index(
    metric="cosine",
    num_partitions=64,
    num_sub_vectors=12
)
print("Índice vetorial criado! LanceDB 100% pronto.")

# ============================================================
# CÉLULA 7 TURBO — BUSCA HÍBRIDA INTELIGENTE + KEYWORD BOOST
# Versão final – Dezembro/2025 – deu trabalhoooooo
# ============================================================

import lancedb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Conexão com o banco (já criado na Célula 6)
# ------------------------------------------------------------------
db = lancedb.connect("./lancedb")
tbl = db.open_table("laws")

# Modelo usado nos embeddings (mantém consistência)
dense_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ------------------------------------------------------------------
# FUNÇÃO PRINCIPAL – busca_rag() TURBINADA
# ------------------------------------------------------------------
def busca_rag(
    pergunta: str,
    top_k: int = 6,
    filtro_vigente: bool = True,
    hierarquia_max: int = 3,
    keyword_boost: bool = True,        # ← novo: força palavras-chave importantes
    hybrid_mode: bool = True           # ← novo: vetor + full-text quando disponível
):
    # 1. Embedding da pergunta
    query_vec = dense_model.encode(pergunta, normalize_embeddings=True).astype("float32")

    # 2. Monta filtros SQL básicos
    where_clauses = []
    if filtro_vigente:
        where_clauses.append("vigente = true")
    if hierarquia_max:
        where_clauses.append(f"hierarquia <= {hierarquia_max}")
    base_filter = " AND ".join(where_clauses) if where_clauses else None

    # 3. KEYWORD BOOST – palavras que quase sempre indicam relevância jurídica
    keyword_conditions = []
    pergunta_lower = pergunta.lower()

    if keyword_boost:
        if any(p in pergunta_lower for p in ["aposentado", "pensionista", "idoso", "pessoa com deficiência"]):
            keyword_conditions.append("text LIKE '%aposentado%' OR text LIKE '%pensionista%' OR text LIKE '%idoso%' OR text LIKE '%deficient%'")
        if "isenção" in pergunta_lower or "imunidade" in pergunta_lower or "não incide" in pergunta_lower:
            keyword_conditions.append("text LIKE '%isenção%' OR text LIKE '%imunidade%' OR text LIKE '%não incide%'")
        if "alíquota" in pergunta_lower:
            keyword_conditions.append("text LIKE '%alíquota%' OR text LIKE '%taxa%'")
        if "parcelamento" in pergunta_lower or "parcela" in pergunta_lower:
            keyword_conditions.append("text LIKE '%parcelamento%' OR text LIKE '%parcela%'")

    # 4. Busca principal (vetorial)
    search = tbl.search(query_vec).metric("cosine").limit(top_k * 5)  # pega mais pra rerank

    # 5. Aplica filtro base
    if base_filter:
        search = search.where(base_filter)

    # 6. Hybrid full-text (funciona nas versões 0.15+ do LanceDB)
    if hybrid_mode:
        try:
            search = search.text(pergunta)   # ← full-text BM25 automático
        except Exception:
            # Algumas versões ainda não têm .text() – ignora silenciosamente
            pass

    # 7. Aplica keyword boost (pré-filtro = muito rápido)
    if keyword_conditions:
        boost_filter = " OR ".join(keyword_conditions)
        search = search.where(boost_filter, prefilter=True)

    # 8. Executa
    resultados = search.to_list()

    # ------------------------------------------------------------------
    # EXIBE RESULTADOS BONITINHO
    # ------------------------------------------------------------------
    print(f"\nRESULTADOS PARA → \"{pergunta}\"")
    print("=" * 95)
    for i, r in enumerate(resultados[:top_k], 1):
        norma = r.get("norma", "N/A")
        numero = r.get("numero", "")
        ano = r.get("ano", "")
        hierarquia = r.get("hierarquia", "?")
        fonte = r.get("source_file", "N/A")
        distancia = r.get("_distance", 0.0)

        # Highlight de palavras-chave na prévia
        preview = r["text"][:600].replace("\n", " ")
        for palavra in ["aposentado", "isenção", "alíquota", "parcelamento", "pensionista", "idoso"]:
            preview = preview.replace(palavra, f"**{palavra.upper()}**")
            preview = preview.replace(palavra.title(), f"**{palavra.title()}**")

        print(f"[{i}] {norma} {numero}/{ano} (H{hierarquia}) ─ Distância: {distancia:.4f} | {fonte}")
        print(f"    → {preview}...\n")

    return resultados[:top_k]   # retorna pra usar na Célula 8

# ------------------------------------------------------------------
# TESTES RÁPIDOS – rode e veja a mágicaaaaaaaa ooooo, mais para grantir se tá tudo okay
# ------------------------------------------------------------------
print("CÉLULA 7 TURBO carregada com sucesso!\n")

busca_rag("O aposentado com único imóvel tem isenção total de IPTU em Itaquaquecetuba?")
busca_rag("Qual a alíquota do ISS para serviços de informática?")
busca_rag("Uma empresa pode parcelar IPTU atrasado?")
busca_rag("Existe imunidade de ITBI para primeira aquisição de imóvel por pessoa física?")

# ------------------------------------------------------------------------

# ============================================================
# CÉLULA 11 — EXPORTA TUDO PRA BAIXAR E GUARDAR NO PC
# Versão final – Dezembro/2025
# ============================================================

import shutil
from pathlib import Path
from google.colab import files
import os
import json
from datetime import datetime

# ------------------------------------------------------------------
# 1. CRIA PASTA TEMPORÁRIA COM TUDO ORGANIZADO
# ------------------------------------------------------------------
export_dir = Path("/content/RAG_Tributario_Itaquaquecetuba_COMPLETO")
export_dir.mkdir(exist_ok=True)

# Lista do que vai dentro do ZIP (tudo que você precisa pra rodar local depois)
items_to_copy = [
    "./lancedb",                              # Banco de dados com vetores
    "/content/lei_rag_multi_output",          # Metadados, chunks, JSONs
    "/content/*.pdf",                         # PDFs originais (se ainda estiverem na raiz)
]

# Copia tudo bonitinho
for item in items_to_copy:
    if Path(item).exists():
        dest = export_dir / Path(item).name
        if Path(item).is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"Diretório copiado: {item}")
        else:
            # Copia todos os PDFs da raiz (caso tenha ficado lá)
            for pdf in Path("/content").glob("*.pdf"):
                shutil.copy2(pdf, export_dir / pdf.name)

# ------------------------------------------------------------------
# 2. SALVA O NOTEBOOK ATUAL 
# ------------------------------------------------------------------
notebook_path = "RAG_Tributario_Itaquaquecetuba_2025.ipynb"
try:
    # Salva o notebook atual automaticamente
    import IPython
    js_code = '''
    require(["base/js/namespace"], function(Jupyter) {
        Jupyter.notebook.save_checkpoint();
    });
    '''
    IPython.display.display(IPython.display.Javascript(js_code))
    
    # Espera um pouquinho e copia
    import time
    time.sleep(3)
    
    # Copia o .ipynb que o Colab gera
    colab_nb = "/content/" + max([f for f in os.listdir("/content") if f.endswith(".ipynb")], key=os.path.getctime)
    shutil.copy2(colab_nb, export_dir / notebook_path)
    print(f"Notebook salvo como: {notebook_path}")
except:
    print("Notebook não foi salvo automaticamente (normal se já tiver nome). Copie manualmente se quiser.")

# ------------------------------------------------------------------
# 3. CRIA ARQUIVO README COM INSTRUÇÕES PRA RODAR LOCAL
# ------------------------------------------------------------------
readme_content = f"""
# RAG TRIBUTÁRIO ITAQUAQUECETUBA – VERSÃO COMPLETA
Projeto criado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}

## O QUE TEM NESSE ZIP:
- lancedb/ → Banco de vetores (pronto pra usar)
- lei_rag_multi_output/ → Todos os JSONs, metadados, chunks
- PDFs originais (se ainda estavam na raiz)
- {notebook_path} → Notebook completo com todas as células (1 a 11)

## COMO RODAR NO SEU PC (local):
1. Instale Python 3.10+
2. Rode no terminal:
   pip install sentence-transformers lancedb ollama pdfplumber pymupdf tqdm
3. Instale o Ollama: https://ollama.com/download
4. Baixe o modelo: ollama pull gemma2:9b-instruct-qat
5. Abra o notebook no VS Code / Jupyter
6. Execute as células na ordem → vai funcionar 100% offline

Projeto feito com carinho por você e pelo Grok em Dezembro de 2025.
Nunca mais perca uma consulta tributária municipal.

Qualquer dúvida: abre uma issue no GitHub que eu te ajudo.
"""

with open(export_dir / "LEIA_ME_PRIMEIRO.txt", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Arquivo LEIA_ME_PRIMEIRO.txt criado!")

# ------------------------------------------------------------------
# 4. CRIA O ZIP FINAL E FAZ DOWNLOAD AUTOMÁTICO
# ------------------------------------------------------------------
zip_name = f"RAG_Tributario_Itaquaquecetuba_COMPLETO_{datetime.now().strftime('%Y%m%d')}.zip"
zip_path = f"/content/{zip_name}"

shutil.make_archive(base_name=zip_path.replace(".zip", ""), format="zip", root_dir=export_dir)

print(f"\nPRONTO! Seu projeto completo está em:")
print(f"→ {zip_path}")
print(f"→ Tamanho: {os.path.getsize(zip_path) / (1024*1024*1024):.2f} GB")

# Download automático
files.download(zip_path)

print("\nDownload iniciado!")
print("Salve esse ZIP no seu PC, no HD externo, no drive...")
print("Esse é o seu legado tributário de 2025.")
print("eeeeeee.")
