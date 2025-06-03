import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA

qa_chain = None

def process_pdf(file):
    global qa_chain
    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm = OpenAI(),
        retriever = retriever,
        return_source_documents = False
    )
    return "PDF processed, You can now ask questions."

def answer_question(question):
    if not qa_chain:
        return "Please upload and process a PDF first."
    return qa_chain.run(question)

with gr.Blocks() as demo:
    gr.Markdown("# PDF chatbot with langchain + Gradio")
    file_input = gr.File(label="Upload PDF")
    status = gr.Textbox(label="Status", interactive=False)
    question = gr.Textbox(label="Your Question")
    answer = gr.Textbox(label="Answer", interactive=False)
    upload_btn = gr.Button("Process PDF")
    ask_btn = gr.Button("Ask") 

    upload_btn.click(process_pdf, inputs=file_input, outputs=status)
    ask_btn.click(answer_question, inputs=question, outputs=answer)

demo.launch()
