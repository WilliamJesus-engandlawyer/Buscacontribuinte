
-----

# 📚 RAG Jurídico Híbrido (LanceDB & Sentence Transformers)

## ⚖️ Descrição do Projeto

Este projeto implementa um **Pipeline RAG (Retrieval-Augmented Generation) Híbrido** especializado na indexação e recuperação inteligente de informações a partir de **documentos legais** (leis, decretos, códigos) e **conceitos jurídicos básicos**.

Ele combina as forças da **Busca Vetorial (Semantic Search)** para entender o *sentido* da pergunta, e a **Busca Full-Text (BM25)** para garantir a precisão das palavras-chave, resultando em uma recuperação de contexto (*retrieval*) altamente relevante para o domínio jurídico.

### 🎯 Casos de Uso

  * **Consultoria Rápida:** Encontrar o artigo de lei ou conceito mais relevante para uma pergunta complexa.
  * **Análise de Processos:** Criar uma base de conhecimento para auxiliar na análise e classificação de processos administrativos/tributários (conforme sugerido no bloco de código de extensão).
  * **Educação Jurídica:** Criar um "cérebro" de conceitos básicos para garantir que a IA entenda a terminologia fundamental.

-----

## 🛠️ Tecnologias Principais

| Categoria | Tecnologia | Função no Pipeline |
| :--- | :--- | :--- |
| **Banco de Dados** | **LanceDB v0.13** | Banco de dados vetorial de código aberto, usado para armazenar os *chunks* de lei e seus *embeddings*, suportando busca híbrida (vetorial + FTS). |
| **Embeddings** | **`intfloat/multilingual-e5-large-instruct`** | Modelo de *embedding* de ponta, otimizado para vários idiomas, superando modelos antigos em português jurídico para gerar vetores de alta qualidade. |
| **Processamento de Texto** | **`pdfplumber`** | Extração do texto dos documentos PDF carregados. |
| **Chunking** | **Regex (Função `extrai_artigos_incisos`)** | Método cirúrgico de segmentação de texto, focado em manter a integridade dos artigos, parágrafos e incisos das leis. |
| **Busca Híbrida** | **BM25 (Nat. no LanceDB & `rank_bm25`)** | Utilizado para a busca por palavras-chave (*full-text search*) e combinação de *scores* na função de consulta (`pergunta`). |
| **Reranking** | **Não especificado no código** | O código inclui uma etapa de `Reranking final (Cross-Encoder)`, mas a importação do `reranker` está faltando. É a etapa final para refinar a ordem dos documentos recuperados. |

-----

## 🚀 Estrutura do Pipeline (Células)

### Célula 1: Instalação e Setup

Instala as bibliotecas necessárias, dando ênfase na instalação forçada do `numpy` para evitar conflitos comuns no ambiente de *notebooks* e garantir compatibilidade.

### Célula 2: Upload, Conceitos e Modelo de Embeddings

1.  Permite o *upload* dos PDFs de leis (ex: CF, CTN, LAI, Lei 9.784).
2.  Define um conjunto de **mais de 70 conceitos jurídicos básicos** (IPTU, Taxa, Princípios) para enriquecer a base de conhecimento e garantir que o RAG tenha uma fundação conceitual sólida.
3.  Carrega o modelo de *embedding* `intfloat/multilingual-e5-large-instruct` na GPU (`device="cuda"`).

### Célula 3: Processamento, Chunking e Criação do LanceDB

Esta é a etapa central:

1.  **Chunking Jurídico:** A função `extrai_artigos_incisos` utiliza *regex* para fatiar o texto, garantindo que cada *chunk* respeite a hierarquia normativa (Artigo, Parágrafo, Inciso).
2.  **Geração de Metadados:** Extrai o nome da norma, o artigo, a fonte (nome do PDF) e classifica como `lei` ou `conceito`.
3.  **Vetorização:** Gera os *embeddings* (vetores de 1024 dimensões) para todos os *chunks*.
4.  **LanceDB:** Cria a tabela `leis` no banco de dados **LanceDB 2.0** (`./lancedb_rag2`), armazena os dados vetoriais e metadados.
5.  **Indexação Híbrida:** Cria índices vetoriais (`cosine` com **IVF\_PQ**) e um índice de busca *full-text* (**FTS/BM25**) para garantir uma recuperação rápida e precisa de ambos os tipos.

### Célula 4: Função de Busca Híbrida e Reranking

A função `pergunta` é a interface de consulta e implementa a estratégia de busca híbrida e reranking:

1.  **Busca Vetorial (k-NN):** Consulta inicial ao LanceDB para encontrar documentos semanticamente similares.
2.  **Refinamento com BM25:** Aplica a pontuação BM25 para as palavras-chave na sub-amostra recuperada.
3.  **Combinação de Scores:** Utiliza pesos ajustáveis (`VETOR_WEIGHT=0.7`, `BM25_WEIGHT=0.3`) para criar um `score_hybrid`, balanceando similaridade semântica e relevância de palavras.
    $$\text{Score Híbrido} = (\text{Vetor Weight} \times \text{Similaridade}) + (\text{BM25 Weight} \times \text{BM25 Score})$$
4.  **Reranking (Cross-Encoder):** (Falta a definição do modelo `reranker`). Um passo final para reordenar os resultados com base em uma análise mais profunda da relevância entre a pergunta e o documento.

-----

## 💡 Próximos Passos Sugeridos

O código já sugere uma excelente evolução:

  * **Tabela `processos` no LanceDB:** Estender a funcionalidade para indexar e buscar resumos e metadados de processos administrativos/tributários (usando o tipo `pa.timestamp` para datas).
  * **Integração com LLM (Geração):** O código atual foca no *Retrieval*. O próximo passo é integrar um LLM (ex: GPT-4, Llama 3) para que, após recuperar os documentos relevantes, ele **gere a resposta** com base no contexto encontrado (RAG completo).
  * **Definição do Reranker:** Incluir o carregamento de um modelo `Cross-Encoder` otimizado para *reranking* (ex: `cross-encoder/ms-marco-TinyBERT-L-2-v2`).

-----

## ⌨️ Como Executar

1.  Abra o arquivo (`.ipynb` ou `.py`) em um ambiente com suporte a GPU (ex: Google Colab).
2.  Execute as células em ordem.
3.  Faça o *upload* dos PDFs de leis quando solicitado.
4.  Utilize a função `pergunta()` para testar a busca:
    ```python
    pergunta("o que é IPTU?")
    ```

-----
