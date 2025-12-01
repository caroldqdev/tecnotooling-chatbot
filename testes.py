from rag import get_embedding, collection_embeddings
import numpy as np
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test_similarity(query):
    query_emb = await get_embedding(query)
    results = []
    async for doc in collection_embeddings.find():
        doc_emb = np.array(doc["embedding"])
        similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb)*np.linalg.norm(doc_emb)+1e-10)
        results.append((similarity, doc["text"]))
    results.sort(key=lambda x: x[0], reverse=True)
    for sim, text in results[:3]:
        print(sim, text[:100])

asyncio.run(test_similarity("Como faço login?"))
