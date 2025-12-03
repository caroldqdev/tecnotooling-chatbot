import os
import numpy as np
import logging
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import cohere
from groq import Groq
import httpx

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

# Clientes
co = cohere.Client(COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client())

# Mongo
mongo = AsyncIOMotorClient(MONGO_URI)
db = mongo["file_data"]
collection = db["embeddings"]

# Modelos
MODEL_EMBED = "embed-multilingual-v2.0"
MODEL_LLM = "groq/compound-mini"

# Hiperparâmetros RAG
TOP_K = 8
MIN_SCORE = 0.25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG")


# ========================================================
# 0) SAUDAÇÃO ESPECIAL DO TOO
# ========================================================
def is_greeting(text: str) -> bool:
    greetings = [
        "oi", "olá", "ola", "oie", "hello", "hi", "hey",
        "bom dia", "boa tarde", "boa noite", "tudo bem"
    ]
    return any(text.lower().startswith(g) for g in greetings)


def too_greeting():
    return {
        "response": (
            "Olá! Eu sou o **Too**, assistente virtual da **Tecnotooling** 🤖✨\n"
            "Estou aqui para te ajudar a consultar documentos internos, responder dúvidas "
            "e facilitar seu dia. Como posso te ajudar hoje?"
        ),
        "paths": []
    }


# ========================================================
# 1) EMBEDDINGS NORMALIZADOS (ouro do RAG)
# ========================================================
async def get_embedding(text: str):
    try:
        res = co.embed(texts=[text], model=MODEL_EMBED)
        emb = np.array(res.embeddings[0])
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb
    except Exception as e:
        logger.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(768)


# ========================================================
# 2) BUSCA MANUAL (cosine similarity)
# ========================================================
async def search_similar_docs(query_emb):
    docs = []

    try:
        async for doc in collection.find({}):

            doc_emb = np.array(doc["embedding"])
            doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-10)

            score = float(np.dot(query_emb, doc_emb))

            docs.append({
                "score": score,
                "text": doc["text"],
                "path": doc.get("path", "Caminho não informado"),
                "id": str(doc.get("_id"))
            })

        docs.sort(key=lambda x: x["score"], reverse=True)

        filtered = [d for d in docs[:TOP_K] if d["score"] >= MIN_SCORE]

        logger.info(f"Docs relevantes encontrados: {[round(d['score'], 4) for d in filtered]}")

        return filtered

    except Exception as e:
        logger.error(f"Erro na busca vetorial manual: {e}")
        return []


# ========================================================
# 3) CONSTRUÇÃO DO CONTEXTO
# ========================================================
def build_context(docs):
    if not docs:
        return "NENHUM DOCUMENTO ENCONTRADO."

    blocks = []
    for i, d in enumerate(docs):
        blocks.append(
            f"### Documento {i+1}\n"
            f"Caminho: {d['path']}\n"
            f"Score: {d['score']:.4f}\n"
            f"Conteúdo:\n{d['text']}\n"
        )
    return "\n".join(blocks)


# ========================================================
# 4) GERAR RESPOSTA COM GROQ + ANTI-ALUCINAÇÃO
# ========================================================
async def llm_answer(query, context):

    prompt = f"""
Você é **Too**, o assistente virtual da Tecnotooling.
Use APENAS o conteúdo dos documentos do contexto.

SE A INFORMAÇÃO NÃO ESTIVER NO CONTEXTO:
responda exatamente:
"Não encontrei essa informação no banco de dados."

---

📚 CONTEXTO:
{context}

❓ PERGUNTA:
{query}

Responda de forma objetiva e, se possível, cite o documento onde encontrou a resposta.
"""

    try:
        resp = groq_client.chat.completions.create(
            model=MODEL_LLM,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "Você é Too, o assistente da Tecnotooling."},
                {"role": "user", "content": prompt}
            ],
        )

        msg = resp.choices[0].message
        txt = msg.content if hasattr(msg, "content") else str(msg)

        return txt

    except Exception as e:
        logger.error(f"Erro ao gerar resposta LLM: {e}")
        return "Erro ao gerar resposta."


# ========================================================
# 5) PIPELINE RAG COMPLETO
# ========================================================
async def rag_answer(query: str):

    # 0) Caso seja saudação, devolve apresentação do Too
    if is_greeting(query):
        return too_greeting()

    # 1) embedding da query
    query_emb = await get_embedding(query)

    # 2) documentos similares
    docs = await search_similar_docs(query_emb)

    # 3) gera contexto
    context = build_context(docs)

    # 4) resposta final via LLM
    answer = await llm_answer(query, context)

    # 5) lista de caminhos
    paths = [d["path"] for d in docs] if docs else []

    return {
        "response": answer,
        "paths": paths
    }
