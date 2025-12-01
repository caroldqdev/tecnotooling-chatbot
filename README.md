# TOO - API de Onboarding (Back-end)

Este repositório contém o back-end (API) para o projeto de onboarding de novos colaboradores, "TOO".

A API é construída em **Python** usando **FastAPI** e liga-se a uma base de dados **MongoDB** para gerir utilizadores e, futuramente, interagir com o sistema de IA (RAG).

---

## Arquitetura e Tecnologias

Este projeto é composto por três partes principais:

1.  **Front-end:** Uma aplicação mobile em React Native (JSX) que consome esta API.
2.  **Back-end (Este Repositório):** Uma API em Python.
    * **FastAPI:** Para criar os endpoints da API de forma rápida e assíncrona.
    * **Motor:** O driver assíncrono para ligar ao MongoDB.
    * **Pydantic:** Para validação de dados (ex: `EmailStr`).
    * **Uvicorn:** Para correr o servidor localmente.
3.  **Base de Dados:**
    * **MongoDB (Atlas):** A nossa base de dados principal na nuvem.
    * **MongoDB (Local):** Usado para testes de desenvolvimento.



---

## Configuração do Ambiente (Desenvolvimento)

Para rodar este projeto localmente, siga estes passos:

### 1. Pré-requisitos

* Python 3.10+
* `pip` (Python package installer)
* Uma conta MongoDB Atlas (para a nuvem) OU uma instalação local do MongoDB Server.

### 2. Instalação

1.  Clona este repositório:
    ```bash
    git clone [URL-DO-TEU-REPOSITORIO-GIT]
    cd too-api
    ```

2.  Cria e ativa um ambiente virtual (venv):
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Instala as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuração do `.env`

1.  Cria um ficheiro chamado `.env` na raiz do projeto.
2.  Adiciona a tua *connection string* do MongoDB.

    **Para ligar ao Atlas (Nuvem):**
    ```.env
    MONGO_URI="mongodb+srv://teu_user:tua_password@teu_cluster.mongodb.net/"
    ```

    **Para ligar ao MongoDB Local:**
    ```.env
    MONGO_URI="mongodb://127.0.0.1:27017"
    ```

### 4. Executar a API

Com o `venv` ativo, corre o servidor Uvicorn:

```bash
uvicorn main:app --reload
