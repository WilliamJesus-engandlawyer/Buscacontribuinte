# ⚖️ RAG Jurídico Inteligente

Sistema completo de **indexação, classificação, chunking e vetorização de normas jurídicas brasileiras**, utilizando **LanceDB**, **Sentence Transformers** e análise de PDF com **pdfplumber**.  
O projeto foi desenvolvido para rodar no **Google Colab**, funcionando como um pipeline RAG educacional e prático.

> 🖤 Há um pequeno easter-egg escondido neste repositório. Nada chamativo… apenas para quem observa o código com atenção.

---

# 📌 Objetivo do Projeto

Criar um pipeline automatizado capaz de transformar coleções de PDFs jurídicos (leis, decretos, constituições, códigos, CTM, CTN, LAI etc.) em um **banco vetorial robusto**, pronto para consultas inteligentes através de modelos de linguagem (LLMs).

Este projeto permite:

- Montar ambientes RAG jurídicos rapidamente  
- Organizar grandes quantidades de documentos legais  
- Criar sistemas de resposta fundamentada  
- Potencializar pesquisas e análises com IA  

---

# 🧠 Funcionalidades

✔ Upload múltiplo de PDFs  
✔ Extração de texto via pdfplumber  
✔ Classificação automática: *Direito Formal* vs *Direito Material*  
✔ Detecção da norma (Lei nº XXXX/AAAA)  
✔ Chunking com overlap  
✔ Geração de embeddings semânticos  
✔ Armazenamento vetorial com LanceDB  
✔ Indexação por similaridade  
✔ Criação de metadados: norma, vigência, categoria e hierarquia  
✔ Totalmente executável no Colab  

---

# 🛠️ Tecnologias Utilizadas

| Tecnologia | Função |
|-----------|--------|
| **LanceDB** | Banco vetorial local e rápido |
| **Sentence Transformers** | Modelo para embeddings |
| **neuralmind/bert-base-portuguese-cased** | BERT especializado em português |
| **pdfplumber** | Extração precisa de texto de PDFs |
| **PyArrow** | Tabelas colunares de alto desempenho |
| **Regex** | Identificação automática de normas |
| **TQDM** | Barras de progresso |
| **Google Colab** | Ambiente de execução |

---

# 🧩 Arquitetura do Pipeline

