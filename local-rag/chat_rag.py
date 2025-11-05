import os
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- Embeddings e armazenamento vetorial ---
def build_vector_store():
    with open("docs/guia_ia.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectordb.persist()
    print("✅ Base vetorial criada.")
    return vectordb


def load_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)


# --- Consulta e geração ---
def retrieve_context(vectordb, query):
    docs = vectordb.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])


def stream_chat(prompt, context):
    payload = {
        "model": "mistral",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um assistente técnico e especializado em IA. "
                    "Use o contexto abaixo para responder.\n\n"
                    f"Contexto:\n{context}"
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        print("\nLLM: ", end="", flush=True)
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                decoded = decoded[len("data: "):]
            try:
                data = json.loads(decoded)

                # 🔍 Ollama pode enviar o texto em dois formatos:
                # 1️⃣ data["message"]["content"]  (novo formato)
                # 2️⃣ data["delta"]["content"]    (antigo formato)
                if "message" in data and "content" in data["message"]:
                    print(data["message"]["content"], end="", flush=True)
                elif "delta" in data and "content" in data["delta"]:
                    print(data["delta"]["content"], end="", flush=True)

            except json.JSONDecodeError:
                continue
        print("\n")


if __name__ == "__main__":
    # cria ou carrega o banco vetorial
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        vectordb = build_vector_store()
    else:
        vectordb = load_vector_store()

    print("💬 Chat com RAG + LLM local (Ollama)\n")
    while True:
        query = input("Você: ")
        if query.lower() in ["sair", "exit", "quit"]:
            break

        context = retrieve_context(vectordb, query)
        stream_chat(query, context)
