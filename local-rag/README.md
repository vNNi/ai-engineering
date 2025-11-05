# RAG

Estudando como funciona de forma simplista uma arquitetura RAG (Retrieval-Augmented Generation - Geração aumentada por recuperação).

É uma arquitetura de IA (ou padrão de projeto) usada para melhorar LLMs.
Ela resolve uma limitação fundamental das LLMs:

LLMs não sabem nada além do que foi treinado nelas.

Nesse exemplo, antes de iniciar a conversa com a LLM via chat no terminal, a aplicação constrói um banco de dados usando "ChromaDB", pegando os dados de um PDF chamado `guia_ia.txt`, onde nele contém informações únicas para nosso software.

E a cada interação do usuário com o chat, nós utilizamos da arquitetura RAG para buscar nesse banco de dados criado com as informações do PDF para dar mais contexto à LLM sobre aquele prompt;

## Technologies

1. LangChain framework - ChromaDB, RecursiveCharacterTextSplitter,  SentenceTransformerEmbeddings
2. Ollama Local model (Mistral)

## Running

### Python

Instale uma versão do Python:

```sh
pyenv virtualenv 3.11.9 local_rag
pyenv virtualenv activate llm-study
```
Instale as dependencias python

```sh
pip install -r requirements.txt
```
### Model

Caso ainda não tenha [Ollma](https://ollama.com/) localmente, instale.

E rode os seguintes comandos:

```sh
ollama serve
```

E rode o modelo desejado:

```sh
ollama run mistral 
```

### Chat

```sh
python chat_rag.py
```

OBS: Aqui talvez precise criar uma pasta chamada `chroma_db`.

E digitar:

```sh
> Qual a senha do universo?
```