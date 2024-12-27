# QA with Retrieval Augmented Generation (RAG)

In this project, two RAG systems were implemented to perform question-answering. The first system used langchain, the second was built from scratch with the assistance of FAISS.

Before using you will need to acquire a HuggingFace API token and add it to line 18 in main.py.

### QA with no RAG

	  python main.py --questions questions.csv --output predictions_no_rag.csv

### QA using RAG with LangChain

	  python main.py --questions questions.csv --rag --langchain --passages passages.csv --output predictions_rag_langchain.csv

### QA using RAG with custom embeddings

	python main.py --questions questions.csv --rag --passages passages.csv --output predictions_rag.csv





