# rag_with_cohere_chat.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import cohere
import asyncio

load_dotenv()

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not COHERE_API_KEY or not MONGO_URI:
    raise Exception("Erro: Uma ou mais chaves de API não foram encontradas.")

# -----------------------------
# CLIENTES
# -----------------------------
co = cohere.Client(COHERE_API_KEY)
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

MODEL_LLM = "command-xlarge-nightly"  # exemplo de LLM Cohere
MODEL_EMBED = "embed-multilingual-v2.0"
EMBED_DIM = 768

# -----------------------------
# FUNÇÃO 1: Gerar embedding
# -----------------------------
async def get_embedding(text: str):
    try:
        response = co.embed(texts=[text], model=MODEL_EMBED)
        emb = np.array(response.embeddings[0], dtype=np.float32)
        if emb.shape[0] != EMBED_DIM:
            logging.warning(f"Embedding inesperado ({emb.shape[0]} dims), ajustando para {EMBED_DIM}")
            emb = np.resize(emb, EMBED_DIM)
        return emb
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(EMBED_DIM, dtype=np.float32)

# -----------------------------
# FUNÇÃO 2: Buscar documentos similares
# -----------------------------
async def search_similar_docs(query_embedding, top_k=None):
    try:
        docs = await collection_embeddings.find().to_list(length=None)
        if not docs:
            return "Nenhum documento encontrado na base."

        embeddings = np.array([np.array(doc["embedding"], dtype=np.float32) for doc in docs])
        texts = [doc["text"] for doc in docs]
        file_names = [doc.get("file_name", f"doc_{i}") for i, doc in enumerate(docs)]

        # Similaridade cosseno
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        sims = (embeddings @ query_embedding) / norms

        # Se top_k não definido, pega todos documentos
        if top_k is None:
            top_indices = np.argsort(sims)[::-1]
        else:
            top_indices = np.argsort(sims)[::-1][:top_k]

        seen_files = set()
        top_texts = []
        for idx in top_indices:
            fname = file_names[idx]
            if fname not in seen_files:
                top_texts.append(f"[{fname}] {texts[idx]}")
                seen_files.add(fname)

        logging.info(f"Total de documentos retornados: {len(top_texts)}")
        return "\n".join(top_texts)

    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# -----------------------------
# FUNÇÃO 3: Gerar resposta com RAG + histórico
# -----------------------------
async def rag_answer(query: str, history=None, top_k=None):
    """
    query: pergunta atual do usuário
    history: lista de mensagens anteriores, ex:
        [
            {"role": "user", "content": "Pergunta anterior"},
            {"role": "assistant", "content": "Resposta anterior"}
        ]
    top_k: quantos documentos buscar
    """
    if history is None:
        history = []

    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb, top_k=top_k)

    # Prompt principal
    prompt = f"""
Você é o assistente Too da TecnoTooling.

Sempre que usar informação de algum documento, cite o nome do arquivo entre colchetes.
Use o contexto completo abaixo para responder à pergunta.

CONTEXTO RELEVANTE:
{context}

PERGUNTA ATUAL:
{query}
"""

    logging.info(f"Prompt enviado ao Cohere Chat (primeiros 500 chars):\n{prompt[:500]}")

    try:
        # Construir histórico + pergunta atual
        inputs = []
        for msg in history:
            role = msg.get("role", "user")
            inputs.append(f"{role.upper()}: {msg['content']}")
        inputs.append(f"USER: {prompt}")

        response = co.chat(
            model=MODEL_LLM,
            inputs=inputs,
            max_output_tokens=600
        )

        # Retorna o conteúdo da primeira resposta
        return response.output[0].content

    except Exception as e:
        logging.error(f"Erro ao gerar resposta com Cohere LLM: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
