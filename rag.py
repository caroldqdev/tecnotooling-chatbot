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

# FUNÇÃO 2: BUSCA HÍBRIDA OTIMIZADA (LITERAL + SEMÂNTICA + CAMINHO) --------
async def search_similar_docs(query_text: str, query_embedding: np.ndarray, k=5):
    results = []
    MIN_SCORE = 0.45

    try:
        query_clean = query_text.strip()

        # ====================================
        # 1️⃣ BUSCA LITERAL FORTE (CNPJ, IT, REG, números, nome do arquivo)
        # ====================================
        literal_hits = []

        async for doc in collection_embeddings.find({
            "$or": [
                {"text": {"$regex": query_clean, "$options": "i"}},
                {"file_name": {"$regex": query_clean, "$options": "i"}},
                {"file_path": {"$regex": query_clean, "$options": "i"}},
                {"text": {"$regex": r"\d{14}", "$options": "i"}}  # padrão CNPJ
            ]
        }):
            literal_hits.append(
                f"📄 Documento: {doc.get('file_name', 'Desconhecido')}\n"
                f"📂 Caminho: {doc.get('file_path', 'Não informado')}\n"
                f"📝 Trecho:\n{doc.get('text', '')[:1200]}\n"
                f"{'-'*60}"
            )

        if literal_hits:
            logging.info("✅ Busca literal encontrou resultados.")
            return "\n".join(literal_hits[:k])

        # ====================================
        # 2️⃣ BUSCA SEMÂNTICA (EMBEDDINGS)
        # ====================================
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        async for doc in collection_embeddings.find(
            {}, {"embedding": 1, "text": 1, "file_name": 1, "file_path": 1}
        ):
            doc_emb = np.array(doc["embedding"])
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)

            similarity = float(np.dot(query_norm, doc_norm))

            if similarity >= MIN_SCORE:
                results.append((
                    similarity,
                    f"📄 Documento: {doc.get('file_name', 'Desconhecido')}\n"
                    f"📂 Caminho: {doc.get('file_path', 'Não informado')}\n"
                    f"📝 Trecho:\n{doc.get('text', '')[:1200]}\n"
                    f"{'-'*60}"
                ))

        results.sort(key=lambda x: x[0], reverse=True)

        top_results = results[:k]

        if not top_results:
            return "❌ Nenhum documento relevante foi encontrado no banco."

        return "\n".join([r[1] for r in top_results])

    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "❌ Falha interna ao buscar documentos."


# -------- FUNÇÃO 3: gerar resposta com RAG (Groq chat) --------
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query,query_emb)

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