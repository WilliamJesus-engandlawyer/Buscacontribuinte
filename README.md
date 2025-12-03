# Diagrama de Arquitetura — RAG Tributário (MVP)

> Documento com diagrama e explicações rápidas para implementar um protótipo de RAG voltado a execução fiscal / citação de contribuinte para uso em prefeitura.

---

```mermaid
flowchart TB
  subgraph UserLayer[Usuário]
    U[Funcionário / Operador] -->|1: Pergunta / Inscrição| Front[Front-end (Streamlit / SPA)]
  end

  subgraph AppLayer[Aplicação]
    Front --> API[API (FastAPI) / Orquestrador]
    API --> Cache[Redis Cache]
    API --> Auth[Auth (JWT / OAuth)]
    API --> QueryDB[SQL DB (Postgres / SQLite) - Cadastros]
    API --> Retrieval[Retrieval Service]
    API --> LLM[LLM API (OpenAI / Groq / Anthropic)]
  end

  subgraph RetrievalLayer[Serviços de Recuperação]
    Retrieval --> VectorIndex[Vector DB (FAISS / Chroma / Qdrant)]
    Retrieval --> DocStore[Document Store (S3 / MinIO) - PDFs/TXTs]
    Retrieval --> Embedding[Embedding Service (sentence-transformers)]
  end

  subgraph Ingest[Ingestão / Indexação]
    Ingester[Ingest Processor]
    Ingester -->|extrai/limpa/chunks| Embedding
    Embedding --> VectorIndex
    Ingester --> DocStore
  end

  subgraph Infra[Infra & Operações]
    VectorIndex --> Backup[Storage (snapshot) / versão]
    QueryDB --> Backup2[Backup SQL]
    API --> Logs[Logs/Auditoria]
    Logs --> SIEM[SIEM / Monitoramento]
    API --> Queue[Fila (RabbitMQ / Redis Queue)]
  end

  %% Flows
  API -->|2: consulta cadastro| QueryDB
  API -->|3: retrieve top-k| Retrieval
  Retrieval -->|4: top-k chunks| API
  API -->|5: monta prompt| LLM
  LLM -->|6: resposta| API
  API -->|7: retorna/gera minuta| Front

  style UserLayer fill:#f9f,stroke:#333,stroke-width:1px
  style AppLayer fill:#efe,stroke:#333,stroke-width:1px
  style RetrievalLayer fill:#eef,stroke:#333,stroke-width:1px
  style Ingest fill:#ffd,stroke:#333,stroke-width:1px
  style Infra fill:#f5f5f5,stroke:#333,stroke-width:1px


---

## Legenda e explicação rápida

* **Front-end**: interface simples (Streamlit para protótipo ou SPA) onde o operador pesquisa por inscrição, CPF/CNPJ ou faz perguntas em linguagem natural.
* **API / Orquestrador**: camada central que coordena a consulta ao banco cadastral e ao service de recuperação; monta prompt e chama o LLM. Recomendo FastAPI (Python) por ser simples e rápido.
* **SQL DB (cadastros)**: armazena os dados transacionais (contribuintes, débitos, histórico). Pode ser SQLite no MVP; Postgres para produção.
* **Retrieval Service**: busca vetorial: consulta FAISS/Chroma/Qdrant para recuperar trechos relevantes da legislação, súmulas e pareceres.
* **Vector DB (FAISS)**: índice vetorial contendo embeddings dos *chunks* de leis, decretos, ementas e pareceres. FAISS funciona bem em CPU para os volumes previstos.
* **Document Store**: S3/MinIO para guardar PDFs originais e versões (útil para auditoria e extração persistente).
* **Embedding Service**: componente responsável por gerar embeddings (p.ex. `sentence-transformers`) durante indexação e para consultas (opcionalmente précomputar apenas para documentos).
* **Ingest Processor**: pipeline de extração de PDF → limpeza → chunk → geração de embedding → indexação. Deve gravar metadados (fonte, vigência, página, data).
* **LLM API**: provedor externo que recebe o prompt com os trechos recuperados e os dados do contribuinte; retorna resposta e, se solicitado, minuta. **Sempre peça para citar fontes** no prompt.
* **Cache (Redis)**: cache de respostas/perguntas frequentes e de prompts montados para reduzir chamadas ao LLM e custos.
* **Fila (Queue)**: para processar tarefas longas (bulk index, geração de minutas em lote, exportação de relatórios).
* **Auth & Logs**: autenticação e auditoria obrigatórias; registre pergunta, trechos usados, usuário, timestamp e resposta para conformidade.

---

## Recomendações de dimensionamento / performance

* **Volumes esperados (prefeitura ~100k habitantes)**

  * Cadastros: 50k–200k registros → Postgres dá conta. Index por inscrição/CPF.
  * Chunks jurídicos: 500–5.000 chunks → FAISS em CPU serve bem.
* **Latência**

  * SQL lookup: < 20 ms
  * FAISS search (10k emb): < 20 ms
  * LLM call: 0.5–2 s (depende do provedor)
* **Reduza custo de LLM**: enviar somente top-3 chunks + resumo dos débitos.

---

## Segurança, LGPD e governança

* Dados pessoais e fiscais são sensíveis — criptografe dados em repouso e em trânsito.
* Autenticação forte (2FA) para operadores que geram minutas/executam atos.
* Mecanismo de revisão humana obrigatório para qualquer ato administrativo/execução.
* Logging e retenção de logs por período acordado (auditoria). Limite quem pode visualizar dados completos.

---

## Passos práticos para implementar o MVP (ordem sugerida)

1. Reunir 10–20 documentos legais (leis municipais + decretos) e 1 CSV de teste (50–200 contribuintes).
2. Implementar ingest pipeline e indexar em FAISS (notebook Colab).
3. Implementar API/Orquestrador: endpoints para `query_by_inscricao` e `query_textual`.
4. Integrar com LLM (OpenAI/Groq) e montar prompt padrão que exige citações de fontes.
5. Criar Streamlit simples com campos: inscrição, pergunta livre, visualizar trechos retornados e gerar minuta.
6. Validar 30 casos reais com equipe jurídica da prefeitura.

---

Se quiser, eu posso gerar também:

* um **diagrama de implantação (infra-as-code)** com serviços Docker/Compose;
* um **notebook Colab** pronto pra indexar 1 PDF e testar 3 queries;
* um **template Streamlit** mínimo integrado ao fluxo.

Diga qual desses você quer que eu gere em seguida.
