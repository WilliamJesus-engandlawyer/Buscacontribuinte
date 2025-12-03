
---
o streamlit não é uma certeza, apenas estou pensando ainda como vai ser essa parte
# 1) Objetivo do MVP (escopo mínimo)

Um serviço que, dada uma pergunta ou um contribuinte:

* Recupera trechos relevantes da legislação municipal/estadual/ federal (normas sobre cobrança, execução fiscal, citação, prescrição);
* Consulta uma tabela (CSV/SQLite) com débitos por contribuinte e retorna se há dívida ativa/valor e status;
* Gera resposta explicativa (ex.: “O contribuinte X está sujeito a execução fiscal por ISS não pago; passos legais: emissão de CDA, citação, prazo etc.”) citando fontes.

O MVP NÃO faz atos processuais automáticos — só gera informação e minutas.

---

# 2) Dados que você precisa (prioridade)

**Essencial (comece aqui)**

1. Texto das leis/ decretos municipais relevantes (impostos municipais: IPTU, ISS, ITBI; normas de cobrança, prazos, prescrição) — em TXT/PDF.
2. CSV com débitos (tabela de contribuintes): id, nome, cpf/cnpj, inscrição, dívida_total, itens (json/relacional), data_última_atualização, status.
3. Ementas / jurisprudência / súmulas (estadual e federal se aplicável) — textos curtos.

**Útil depois**

* Modelos de CDA / certidões / intimações (templates).
* Histórico de cobranças (eventos).
* Doutrina/pareceres da procuradoria municipal (resumos).

---

# 3) Estrutura de arquivos / banco (planejamento ainda pensando)

```
project/
 ├─ data/
 │   ├─ leis/         # pdfs/txt
 │   ├─ jurisprudencia/ # txt
 │   └─ contribuinte_debitos.csv
 ├─ notebooks/
 ├─ src/
 │   ├─ ingest.py
 │   ├─ index_faiss.py
 │   ├─ query.py
 │   └─ app_streamlit.py
 └─ index/            # arquivos FAISS + metadatas.json
```

Exemplo `contribuinte_debitos.csv`:

```csv
id,inscricao,nome,cpf_cnpj,email,endereco,divida_total,itens_json,data_atualizacao,status
1,2023A0001,João Silva,12345678901,joao@ex.com,"Rua X, 10",1500.00,"[{""tipo"":""IPTU"",""valor"":1500.00,""ano"":2023,""cda"":""CDA123""}]",2025-11-01,ativo
2,2020B0055,Empresa Y,01234567000199,contato@y.com,"Av Y, 100",0.00,"[]",2025-02-15,quitado
```

SQL (SQLite) simples para histórico:

```sql
CREATE TABLE contrib_debitos (
  id INTEGER PRIMARY KEY,
  inscricao TEXT,
  nome TEXT,
  cpf_cnpj TEXT,
  email TEXT,
  endereco TEXT,
  divida_total REAL,
  itens_json TEXT,
  data_atualizacao TEXT,
  status TEXT
);
```

---

# 4) Arquitetura proposta (MVP)

1. **Ingestão**: extrair texto de PDFs (pdfplumber/pypdf), limpar, dividir em chunks (parágrafo ou 200–400 palavras).
2. **Embeddings**: `sentence-transformers` (modelo multilíngue leve).
3. **Indexação**: FAISS local (IndexFlatIP + normalização). Guardar metadados (fonte, artigo, página, id).
4. **Banco contribuinte**: CSV/SQLite para buscar dívida atual.
5. **Flow de query**:

   * Usuário pergunta / fornece inscrição do contribuinte.
   * Gera embedding da pergunta.
   * Recupera top-k trechos (k = 3–6).
   * Busca débito no CSV/SQLite pelo contribuinte.
   * Monta prompt com o contexto legal + dados do contribuinte.
   * Chama LLM (API externa) para gerar resposta e, se pedida, minuta de citação/execução.
6. **Interface**: Streamlit ou CLI para testes.

---

# 5) Código mínimo — passo a passo pronto pra Colab/local

Instala:

```bash
pip install sentence-transformers faiss-cpu pdfplumber pandas streamlit
```

Ingestão + chunking (simplificado):

```python
# src/ingest.py
import pdfplumber, re, json, os
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_texts_from_pdfs(folder):
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    docs.append({"source": fname, "page": i+1, "text": text})
    return docs

def chunk_text(text, max_words=250, overlap=50):
    words = re.findall(r'\S+', text)
    chunks = []
    i=0
    while i < len(words):
        chunk = ' '.join(words[i:i+max_words])
        chunks.append(chunk)
        i += max_words - overlap
    return chunks

def build_documents(pdf_folder):
    pages = extract_texts_from_pdfs(pdf_folder)
    documents = []
    doc_id = 0
    for p in pages:
        chunks = chunk_text(p['text'])
        for c in chunks:
            documents.append({
                "id": doc_id,
                "source": p['source'],
                "page": p['page'],
                "text": c
            })
            doc_id += 1
    return documents
```

Gerar embeddings e indexar FAISS:

```python
# src/index_faiss.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss, json

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def index_documents(documents, index_path="index/faiss.index", meta_path="index/metadatas.json"):
    texts = [d['text'] for d in documents]
    embs = model.encode(texts, show_progress_bar=True, batch_size=32)
    embs = np.array(embs).astype('float32')
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(documents, f)
    return index_path, meta_path

def load_index(index_path="index/faiss.index"):
    index = faiss.read_index(index_path)
    return index
```

Recuperação + montar prompt + consulta ao CSV:

```python
# src/query.py
import json, sqlite3, pandas as pd
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_metadata(meta_path="index/metadatas.json"):
    with open(meta_path) as f:
        return json.load(f)

def retrieve(query, index, metas, top_k=4):
    q_emb = model.encode([query]).astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        m = metas[idx]
        results.append({"score": float(score), "meta": m, "text": m['text']})
    return results

def busca_debito_por_inscricao(csv_path, inscricao):
    df = pd.read_csv(csv_path)
    r = df[df['inscricao'] == inscricao]
    if r.empty:
        return None
    row = r.iloc[0].to_dict()
    return row

def build_prompt_for_exec(query, retrieved, debito):
    context = "\n\n".join([f"[Fonte: {r['meta']['source']}|page:{r['meta']['page']}]\n{r['text']}" for r in retrieved])
    debito_text = f"Contribuinte: {debito['nome']} | inscrição: {debito['inscricao']} | dívida_total: {debito['divida_total']} | itens: {debito['itens_json']}" if debito else "Sem registro de débito encontrado."
    prompt = f"""Você é um assistente jurídico da Procuradoria Municipal. Use SOMENTE o contexto abaixo para responder objetivamente.
Contexto legal:
{context}

Dados do contribuinte:
{debito_text}

Pergunta: {query}

Instruções: explique se há fundamento para execução fiscal, passo a passo (emitir CDA, citação, prazo), indique artigos/trechos usados (fonte e página). Se houver débito, gere uma minuta curta de intimação (máx. 200 palavras). Seja objetivo e cite as fontes."""
    return prompt
```

Obs.: para chamar o LLM substitua `call_llm(prompt)` pela sua API (OpenAI/Groq). Sempre limite tokens e peça citação de fontes.

---

# 6) Prompt templates (exemplos prontos)

**Consulta jurídica (execução/citação):**

```
Você é assistente jurídico da Prefeitura. Use apenas o contexto fornecido (cite fontes) e os dados do contribuinte.
Pergunta: <pergunta do usuário>
Responda em português, indicando:
1) Existe fundamento para iniciar execução fiscal? (sim/não) — e por quê.
2) Passos práticos (emitir CDA, publicação, citação) com prazos.
3) Listar os artigos/trechos usados (Fonte: arquivo.pdf | página X).
4) Se houver débito, gerar minuta de intimação (até 200 palavras).
Se não houver débito, informe procedimento administrativo alternativo.
```

**Geração de minuta apenas (use quando já souber que há débito):**

```
Gere uma minuta formal de intimação para cobrança administrativa do débito abaixo.
Dados do contribuinte: <nome, inscricao, endereco, valor, itens>
Regras estilísticas: 1 parágrafo objetivo, linguagem formal, indicar prazo para pagamento e consequência de não pagamento.
```

---

# 7) Métricas simples de validação (jurídico)

* **Precisão das fontes**: em X% das respostas, as fontes citadas devem realmente conter a afirmação.
* **Conformidade**: revisão por um advogado da prefeitura em 20 casos.
* **False Positive** (indicar cobrança onde não há) — crítico. Configure revisão humana obrigatória para execução.

---

# 8) Boas práticas legais e de privacidade (pensar tambem quandoo projeto avançar)

* Dados fiscais são sensíveis: **criptografe o CSV** em produção e restrinja acesso.
* Nunca automatize despacho de atos sem revisão humana.
* Mantenha versão da legislação (vigência): armazene data e vigência nos metadados de cada chunk.
* Logue todas as consultas e respostas para auditoria.

---

# 9) Redução de custo e otimização

* Envie só top-3 chunks para o LLM.
* Use modelos de embedding leves em CPU.
* Quantize index FAISS se crescer muito (IVF/PQ).
* Use prompts curtos e bem-estruturados para reduzir tokens faturados na API.

---

# 10) Interface simples para protótipo

* **Streamlit** pode ser trocado e claro depois, o importante e vir com:

  * input: inscrição ou texto de pergunta
  * painel: resultados recuperados (trechos + fonte)
  * painel: débito do contribuinte (tabela)
  * botão: gerar minuta (chama LLM)
* Fácil de publicar em Heroku/Render (para testes internos) ou rodar local.

---

# 11) Plano, passo iniciais

1. Reúna: 3-5 PDFs de legislação municipal + CSV de 50 registros fictícios (pode gerar aleatório)
2. Rodar o ingest + index (código acima) em Colab/PC — cria FAISS + metadatas
3. Teste retrieval com queries manuais — ajuste chunk size / k
4. Integre com LLM (OpenAI/Groq, não precisa ser openai, mas acho que ela é a melhor) aplicar os prompts acima
5. Monte um Streamlit simples pra demonstrar pro pessoal da prefeitura

---

# 12) Riscos e limites (é um projeto de estudante)

* RAG fornece **assistência**, não substitui decisão jurídica.
* É necessário **avalizar** as minutas e execuções por procurador humano.
* Documente limitações do sistema para evitar uso indevido.

---
