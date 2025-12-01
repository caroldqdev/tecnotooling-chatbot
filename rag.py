# rag.py
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
MODEL_LLM = "groq/compound-mini"  # para chat
MODEL_EMBED = "embed-multilingual-v2.0"  # Cohere embeddings

# -------- FUNÇÃO 1: gerar embedding (Cohere) --------
async def get_embedding(text: str):
    try:
        response = co.embed(texts=[text], model=MODEL_EMBED)
        emb = np.array(response.embeddings[0])
        logging.info(f"Embedding da pergunta ({len(emb)} dims): {emb[:10]} ...")  # primeiros 10 valores
        return emb
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(768)  # fallback

# -------- FUNÇÃO 2: buscar top-k documentos similares --------
async def search_similar_docs(query_embedding, k=3):
    results = []
    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc["embedding"])
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-10
            )
            results.append((similarity, doc["text"]))

        results.sort(key=lambda x: x[0], reverse=True)
        top_texts = [r[1] for r in results[:k]]

        logging.info(f"Top {k} documentos mais similares (scores): {[r[0] for r in results[:k]]}")
        for i, (sim, text) in enumerate(results[:k]):
            logging.info(f"[{i}] Similaridade: {sim:.4f}, Texto: {text[:100]}...")  # primeiros 100 chars

        return "\n".join(top_texts)
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# -------- FUNÇÃO 3: gerar resposta com RAG (Groq chat) --------
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    prompt = f"""
Você é um assistente útil.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma clara e objetiva, sempre trazando respostas que envolvam a Tecnotooling
"""
    logging.info(f"Prompt final enviado para o Groq:\n{prompt[:500]}...")  # primeiros 500 caracteres

    try:
        from groq import Groq
        import httpx
        groq_client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client())

        response = groq_client.chat.completions.create(
            model=MODEL_LLM,
            messages=[
                {"role": "system", "content": "Você se chama Too, o assistente dos colaboradores da TecnoTooling. Você deve ser prestativo e atencioso"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        message_obj = response.choices[0].message
        logging.info(f"Resposta bruta do Groq: {message_obj}")

        if hasattr(message_obj, "content"):
            return message_obj.content
        elif isinstance(message_obj, list):
            return " ".join([m.content for m in message_obj])
        else:
            return str(message_obj)

    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."