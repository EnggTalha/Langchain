import os
from langchain.document_loaders import UnstructuredPDFLoader  # To load unstructured PDF data (not used in current code)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split large text into smaller chunks
from langchain.vectorstores import Chroma  # For document retrieval and vector storage
from langchain.embeddings import OpenAIEmbeddings  # For using OpenAI embeddings
from langchain.llms import OpenAI  # For using the OpenAI language model
from langchain.chains.question_answering import load_qa_chain  # For setting up the QA chain
from PyPDF2 import PdfReader  # To extract text from PDFs

def comp_process(apikey, pdfs, question):
    # Set the OpenAI API key for language model access
    os.environ["OPENAI_API_KEY"] = apikey  # Set environment variable for OpenAI API key
    
    # Initialize OpenAI language model with specific temperature and API key
    llm = OpenAI(temperature=0, openai_api_key=apikey)  # The temperature is set to 0 for deterministic responses

    # Initialize an empty string to store text extracted from all PDF files
    text = ""

    # Loop over all provided PDF files
    for file in pdfs:
        pdf_reader = PdfReader(file)  # Read the PDF file using PdfReader
        # Extract text from each page of the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()  # Append extracted text from each page

    # Initialize the text splitter to break large text into chunks of 1000 characters without overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Set the chunk size and overlap
    chunks = text_splitter.split_text(text=text)  # Split the text into chunks

    # Generate embeddings using OpenAI for the chunks of text
    embeddings = OpenAIEmbeddings(openai_api_key=apikey)  # Initialize OpenAI embeddings with the API key
    # Create a Chroma vector store using the generated embeddings
    docsearch = Chroma.from_texts(chunks, embedding=embeddings).as_retriever()  # Store and retrieve documents based on embeddings

    if question:  # Check if a question is provided
        # Get the relevant documents based on the question from the Chroma retriever
        docs = docsearch.get_relevant_documents(question)
        # Set up the question-answering chain using the language model
        read_chain = load_qa_chain(llm=llm)  # Initialize the QA chain with the language model
        # Run the QA chain on the relevant documents to get the answer
        answer = read_chain.run(input_documents=docs, question=question)  # Get the answer to the question

    return answer  # Return the generated answer
