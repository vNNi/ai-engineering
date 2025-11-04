import requests
import json

url = "http://localhost:11434/api/chat"

def stream_chat(prompt):
    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "Você é um assistente técnico e amigável."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    with requests.post(url, json=payload, stream=True) as response:
        print("\nLLM: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8").strip()
                # Algumas linhas podem começar com "data: "
                if decoded.startswith("data: "):
                    decoded = decoded[len("data: "):]

                try:
                    data = json.loads(decoded)
                    # O campo correto que contém o texto gerado é `content` dentro de `delta`
                    # Nem todas as linhas têm delta, então usamos get
                    if "delta" in data and "content" in data["delta"]:
                        print(data["delta"]["content"], end="", flush=True)
                except json.JSONDecodeError:
                    continue  # ignora linhas não JSON
        print("\n")  # nova linha no final

if __name__ == "__main__":
    print("💬 Chat stream com LLM local (Ollama)\n")
    while True:
        prompt = input("Você: ")
        if prompt.lower() in ["sair", "exit", "quit"]:
            break
        stream_chat(prompt)
