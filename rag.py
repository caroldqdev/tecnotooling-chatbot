import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import cohere
import asyncio

load_dotenv()

# ======================
# CONFIGURAÇÕES
# ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GROQ_API_KEY:
    raise Exception("Erro: GROQ_API_KEY não encontrada.")
if not COHERE_API_KEY:
    raise Exception("Erro: COHERE_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

# Cohere para embeddings
co = cohere.Client(COHERE_API_KEY)

# Mongo
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

MODEL_EMBED = "embed-multilingual-v2.0"
MODEL_LLM = "groq/compound-mini"

TOP_K = 8         # número de documentos recuperados
MIN_SCORE = 0.25  # similaridade mínima para aceitar documento


# ========================================================
# 1) GERA EMBEDDING NORMALIZADO
# ========================================================
async def get_embedding(text: str):
    try:
        response = co.embed(texts=[text], model=MODEL_EMBED)
        emb = np.array(response.embeddings[0])
        # normalização (MUDA TUDO — letra de ouro do RAG)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb

    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(768)


# ========================================================
# 2) BUSCA TOP K DOCUMENTOS + SCORES + CAMINHO
# ========================================================
async def search_similar_docs(query_embedding, k=TOP_K):

    results = []

    try:
        async for doc in collection_embeddings.find():

            doc_emb = np.array(doc["embedding"])
            doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)

            # cosine similarity
            similarity = float(np.dot(query_embedding, doc_emb))

            results.append({
                "score": similarity,
                "text": doc["text"],
                "path": doc.get("path", "caminho_desconhecido"),
                "id": str(doc.get("_id"))
            })

        # ordena por score
        results.sort(key=lambda x: x["score"], reverse=True)

        # aplica score mínimo
        filtered = [r for r in results[:k] if r["score"] >= MIN_SCORE]

        logging.info(f"Top documentos relevantes: {[r['score'] for r in filtered]}")

        return filtered

    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return []


# ========================================================
# 3) GERA RESPOSTA COM RAG + CAMINHOS
# ========================================================
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    docs = await search_similar_docs(query_emb)

    if not docs:
        context_text = "NENHUM DOCUMENTO RELEVANTE FOI ENCONTRADO."
    else:
        context_text = "\n\n".join([
            f"### Documento {i+1}\n"
            f"Caminho: {d['path']}\n"
            f"Score: {d['score']:.4f}\n"
            f"Conteúdo:\n{d['text']}"
            for i, d in enumerate(docs)
        ])

    # PROMPT ANTI-ALUCINAÇÃO (+++)
    prompt = f"""
Você é um assistente da Tecnotooling chamado Too.
Use APENAS as informações encontradas no contexto abaixo.

SE A RESPOSTA NÃO ESTIVER NO CONTEXTO → diga:

"Não encontrei essa informação no banco de dados."

📚 CONTEXTO (DOCUMENTOS ENCONTRADOS):
{context_text}

❓ PERGUNTA:
{query}

Agora responda de forma clara, objetiva e cite de qual documento a resposta veio.
"""

    try:
        from groq import Groq
        import httpx

        groq_client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client())

        response = groq_client.chat.completions.create(
            model=MODEL_LLM,
            messages=[
                {"role": "system", "content": "Você se chama Too e é o assistente dos colaboradores da TecnoTooling."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )

        message_obj = response.choices[0].message

        final_answer = message_obj.content if hasattr(message_obj, "content") else str(message_obj)

        # adiciona caminhos no final da resposta (obrigatório)
        if docs:
            caminhos = "\n".join([f"- {d['path']} (score {d['score']:.4f})" for d in docs])
        else:
            caminhos = "Nenhum documento encontrado."

        return f"{final_answer}\n\n📂 Documentos utilizados:\n{caminhos}"

    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
