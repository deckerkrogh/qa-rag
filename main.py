from argparse import ArgumentParser

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
import faiss
import pickle

from tqdm import tqdm

MODEL = HuggingFaceHub(
        huggingfacehub_api_token='[TOKEN HERE]',
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256}
    )


def eval_chain(chain, questions):
    print("Format is: true answer | model answer\n")
    for i, qa in questions.iterrows():
        print(qa['question'])
        model_ans = chain.invoke(qa['question'])
        print(f"{qa['answer'][2:-2]} | {model_ans}\n")


def langchain_predict(chain, questions, out):
    answers = []
    print("Querying model...")
    for q in tqdm(questions):
        answers.append(chain.invoke(q))
    d = {"question": questions, "answer": answers}
    pd.DataFrame(d).to_csv(out, index=False)


def norag_langchain(q_file, output_file):
    # Simple QA
    questions = pd.read_csv(q_file)["question"].to_list()
    q_prompt = PromptTemplate.from_template("Question: {question}\nAnswer:")
    output_parser = StrOutputParser()
    chain = {"question": RunnablePassthrough()} | q_prompt | MODEL | output_parser
    langchain_predict(chain, questions, output_file)


def rag_langchain(q_file, output_file):
    loader = CSVLoader(passages_file)
    docs = loader.load()
    questions = pd.read_csv(q_file)["question"].to_list()

    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0, separator='\n')
    docs = text_splitter.split_documents(docs)

    # Get string representations of documents, include title and context
    doc_texts = [' '.join(doc.page_content.split("\n")[2:]) for doc in docs]

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(doc_texts, embeddings)
    print("Created embeddings.")

    # RAG with LangChain
    # rag_prompt = PromptTemplate.from_template("Question: {question}\nContext:{context}\nAnswer:")
    rag_prompt = PromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext:{context}\nAnswer:")
    retriever = db.as_retriever(search_kwargs={'k': 3})  # k=4 is default
    chain = {"question": RunnablePassthrough(), "context": retriever} | rag_prompt | MODEL | StrOutputParser()
    langchain_predict(chain, questions, output_file)


def rag(passages_file, q_file, out_file):
    questions = pd.read_csv(q_file)["question"].to_list()
    doc_texts = pd.read_csv(passages_file)["context"].to_list()

    # Save time rerunning with pickling
    embed_model = SentenceTransformer('sentence-transformers/roberta-base-nli-stsb-mean-tokens')
    emb_meta_file = "emb_meta.pickle"
    doc_emb_file = "passages_emb.pickle"
    q_emb_file = "q_emb.pickle"
    try:
        emb_meta = pickle.load(open(emb_meta_file, 'rb'))
        if emb_meta != (passages_file, q_file):
            raise Exception
        doc_embeddings = pickle.load(open(doc_emb_file, 'rb'))
        q_embeddings = pickle.load(open(q_emb_file, 'rb'))
        print("Loaded embeddings from pickle")
    except Exception as e:
        print(f'exception: {e}')
        print("Generating embeddings...")
        doc_embeddings = embed_model.encode(doc_texts)
        q_embeddings = embed_model.encode(questions)
        pickle.dump(doc_embeddings, open(doc_emb_file, 'wb'))
        pickle.dump(q_embeddings, open(q_emb_file, 'wb'))
        pickle.dump((passages_file, q_file), open(emb_meta_file, 'wb'))
        print("Embeddings created\n")

    # Find k-most similar doc indices for each question
    k = 3
    index_obj = faiss.IndexFlatL2(len(doc_embeddings[0]))
    index_obj.add(doc_embeddings)
    _, doc_indices = index_obj.search(q_embeddings, k)

    answers = []
    print("Pinging model...")
    for i, q in enumerate(tqdm(questions)):
        context = ' '.join([doc_texts[doc_indices[i][d]] for d in range(len(doc_indices[i]))])
        prompt = f"Question: {q}\nContext: {context}\nAnswer:"
        answers.append(MODEL.invoke(prompt))
        # answers.append("placeholder")

    # Output answers to csv
    df = pd.DataFrame({"question": questions, "answer": answers})
    df = pd.concat([df, pd.DataFrame(doc_indices)], axis=1)
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser("homework CLI")

    parser.add_argument('--langchain', action="store_true", help="langchain rag")
    parser.add_argument('--rag', action="store_true", help="decker rag system")

    parser.add_argument('--questions', help="path to questions file")
    parser.add_argument('--passages', help="path to passages file")
    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()
    passages_file = args.passages
    q_file = args.questions
    output_file = args.output

    if args.langchain:
        rag_langchain(q_file, output_file)

    elif args.rag:
        rag(passages_file, q_file, output_file)

    else:
        norag_langchain(q_file, output_file)

