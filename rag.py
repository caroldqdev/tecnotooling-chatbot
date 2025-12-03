# rag.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import cohere

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
co = cohere.Client(COHERE_API_KEY)

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# ---- MODELOS ----
MODEL_LLM = "groq/compound-mini"
MODEL_EMBED = "embed-multilingual-v2.0"

# ---- CONTROLE DO RAG ----
TOP_K = 5
MIN_SCORE = 0.25


# =====================================================
# 1) GERA EMBEDDING NORMALIZADO (OTIMIZADO)
# =====================================================
async def get_embedding(text: str):
    try:
        response = co.embed(texts=[text], model=MODEL_EMBED)
        emb = np.array(response.embeddings[0])

        # ✅ NORMALIZAÇÃO (MUITO IMPORTANTE)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(768)


# =====================================================
# 2) BUSCA OTIMIZADA TOP-K POR COSINE SIMILARITY
# (SEM QUEBRAR O FORMATO DO SEU FRONT)
# =====================================================
async def search_similar_docs(query_embedding, k=TOP_K):
    results = []

    try:
        async for doc in collection_embeddings.find({}, {"embedding": 1, "text": 1}):
            doc_emb = np.array(doc["embedding"])

            # ✅ NORMALIZAÇÃO DO DOCUMENTO
            norm = np.linalg.norm(doc_emb)
            if norm > 0:
                doc_emb = doc_emb / norm

            similarity = float(np.dot(query_embedding, doc_emb))

            if similarity >= MIN_SCORE:
                results.append((similarity, doc["text"]))

        # ✅ ORDENA E PEGA TOP-K
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:k]

        return "\n".join([r[1] for r in top_results])

    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."


# =====================================================
# 3) GERA RESPOSTA COM RAG (MANTIDO IGUAL AO SEU)
# =====================================================
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    prompt = f"""
Você é um assistente útil.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma clara e objetiva, sempre trazendo respostas que envolvam a Tecnotooling
"""

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

        if hasattr(message_obj, "content"):
            return message_obj.content
        elif isinstance(message_obj, list):
            return " ".join([m.content for m in message_obj])
        else:
            return str(message_obj)

    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
