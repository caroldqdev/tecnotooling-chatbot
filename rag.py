# rag_optimized.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import cohere
import asyncio

load_dotenv()

# ---- CONFIGURAÇÕES ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GROQ_API_KEY:
    raise Exception("Erro: GROQ_API_KEY não encontrada.")
if not COHERE_API_KEY:
    raise Exception("Erro: COHERE_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

# ---- CLIENTES ----
co = cohere.Client(COHERE_API_KEY)  # Embeddings Cohere

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# Modelos
MODEL_LLM = "groq/compound-mini"
MODEL_EMBED = "embed-multilingual-v2.0"
EMBED_DIM = 768  # garantir 768 dimensões

# -------- FUNÇÃO 1: gerar embedding (Cohere) --------
async def get_embedding(text: str):
    try:
        response = co.embed(texts=[text], model=MODEL_EMBED)
        emb = np.array(response.embeddings[0], dtype=np.float32)
        if emb.shape[0] != EMBED_DIM:
            logging.warning(f"Embedding inesperado ({emb.shape[0]} dims), ajustando para {EMBED_DIM}")
            emb = np.resize(emb, EMBED_DIM)
        logging.info(f"Embedding gerado ({len(emb)} dims)")
        return emb
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(EMBED_DIM, dtype=np.float32)

# -------- FUNÇÃO 2: buscar top-k documentos similares --------
async def search_similar_docs(query_embedding, top_k=5):
    try:
        docs = await collection_embeddings.find().to_list(length=None)
        embeddings = np.array([np.array(doc["embedding"], dtype=np.float32) for doc in docs])
        texts = [doc["text"] for doc in docs]
        file_names = [doc.get("file_name", f"doc_{i}") for i, doc in enumerate(docs)]

        # Similaridade cosseno
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        sims = (embeddings @ query_embedding) / norms

        # Top-k índices
        top_indices = np.argsort(sims)[::-1][:top_k]

        # Agrupar por documento (evita chunks repetidos)
        seen_files = set()
        top_texts = []
        for idx in top_indices:
            fname = file_names[idx]
            if fname not in seen_files:
                top_texts.append(texts[idx])
                seen_files.add(fname)

        logging.info(f"Top {len(top_texts)} documentos retornados")
        return "\n".join(top_texts)

    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# -------- FUNÇÃO 3: gerar resposta com RAG --------
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb, top_k=5)

    prompt = f"""
Você é um assistente útil.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma clara e objetiva, sempre trazendo respostas que envolvam a TecnoTooling
"""
    logging.info(f"Prompt enviado ao Groq:\n{prompt[:500]}...")  # primeiros 500 chars

    try:
        from groq import Groq
        import httpx
        groq_client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client())

        response = groq_client.chat.completions.create(
            model=MODEL_LLM,
            messages=[
                {"role": "system", "content": "Você se chama Too, o assistente dos colaboradores da TecnoTooling. Seja prestativo e atencioso."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        message_obj = response.choices[0].message
        if hasattr(message_obj, "content"):
            return message_obj.content
        elif isinstance(message_obj, list):
            return " ".join([m.content for m in message_obj])
        else:
            return str(message_obj)

    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
