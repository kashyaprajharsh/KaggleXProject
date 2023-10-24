import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
#from PyPDF2 import PdfFileReader
from transformers import AutoTokenizer
#from langchain.llms import CTransformers
#from langchain.llms import Replicate
from langchain.llms import VertexAI
#from langchain.chat_models import ChatVertexAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
#from langchain.llms import HuggingFacePipeline
#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,HypotheticalDocumentEmbedder
#from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import TextLoader
#from langchain.document_loaders import Docx2txtLoader
import os
from index_html import css,bot_template,user_template
import vertexai
import re
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import pandas as pd
from google.cloud import storage


load_dotenv()



def get_pdf_text(pdf_docs):
    load_dotenv()
    text=""
    #Construct the full path to the PDF file
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text



def get_text_chunks(docs):
    load_dotenv()
    
    #tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function = len,
    )
    
    
    chunks = text_splitter.split_text(docs)
    return chunks



def get_vectorstore(chunks):
    load_dotenv()
     # Embedding
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                        model_kwargs={'device': 'cpu'})
    #embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", 
                                         #model_kwargs={'device': 'cpu'})
    #REQUESTS_PER_MINUTE = 150                                   
    #embeddings = VertexAIEmbeddings()
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",model_kwargs={'device': 'cpu'})

    #HYDE
   

    persist_directory = "vector_db"
    vectordb = Chroma.from_texts(texts=chunks,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
    
    vectordb.persist()
    vectordb = None
    vectordb = Chroma(persist_directory=persist_directory,
                       embedding_function=embeddings)
    return vectordb



def get_conversation_cahin(vectordb):
    load_dotenv()

    #llm = ChatVertexAI()
    llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1000,
    temperature=0,
    top_p=0.9,
    top_k=40,
    verbose=True,
    )

    #embeddings = HypotheticalDocumentEmbedder(
    #llm_chain=llm, base_embeddings=base_embeddings
    #)

    #llm = Replicate(
        #streaming=True,
        #model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        #model="meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
        #callbacks=[StreamingStdOutCallbackHandler()],
        #model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1}
         #)
    
     #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                        #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})


    #llm = HuggingFacePipeline.from_model_id(
    #model_id="bigscience/bloom-560m",
    #task="text-generation",
    #model_kwargs={"temperature" : 0, "max_length" : 1024})

    #llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k": 3,"fetch_k": 4,"include_metadata": True},search_type ="mmr"),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain


def handle_userinput(user_question):
    #response = st.session_state.conversation({'question': user_question})
    #chat_history = []
    response = st.session_state.conversation({'question': user_question})
    #st.write(response['answer'])
    #st.write(response['source_documents'][1])

    #with st.sidebar:
        #for chunk in l:
            #st.write(source_template.replace(
                    #"{{MSG}}",chunk.page_content), unsafe_allow_html=True)

    #st.write(response['answer'])
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def list_pdfs_in_folder(folder_path):
    pdf_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(folder_path, filename))
    return pdf_files



def extract_year_from_filename(filename):
    # Use regular expressions to extract the year
    match = re.search(r'(\w{3})(\d{2})', filename)
    
    if match:
        month_abbreviation = match.group(1)
        year_short = match.group(2)
        
        # Define a dictionary to map month abbreviations to their numeric values
        month_mapping = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Convert the month abbreviation to a numeric value
        month_numeric = month_mapping.get(month_abbreviation.lower())
        
        if month_numeric:
            # Convert the year to a four-digit format
            year = '20' + year_short
            return year

    return None

def get_fiscal_quarter(fiscal_year_month):
    return pd.Period(fiscal_year_month, freq="Q").quarter


def get_month_from_filename(filename):
    # Use regular expressions to extract the month abbreviation
    match = re.search(r'(\w{3})\d{2}', filename)
    
    if match:
        return match.group(1).lower()
    
    return None


def folder_selector():
    st.title("Select the Company and the earning calls.")

    # Create a dropdown to select a name
    #selected_name = st.selectbox("Select a name:", ['ADANIENT','ADANIPORTS','APOLLOHOSP','ASIANPAINT','AXISBANK','BAJAJ-AUTO',
                                    #'BAJFINANCE','BAJAJFINSV','BHARTIARTL','BPCL','BRITANNIA','CIPLA','COALINDIA',
                                    #'DIVISLAB','DRREDDY','EICHERMOT','GRASIM',
                                    #'HCLTECH','HDFCBANK','HDFCLIFE','HEROMOTOCO','HINDALCO','HINDUNILVR',
                                    #'ICICIBANK','INDUSINDBK','INFY','ITC','JSWSTEEL','KOTAKBANK','LT','M&M','MARUTI','NESTLEIND','NTPC',
                                    #'ONGC','POWERGRID','RELIANCE','SBILIFE','SBIN','SUNPHARMA',
                                    #'TCS','TATACONSUM','TATAMOTORS','TATASTEEL','TECHM','TITAN','ULTRACEMCO','UPL','WIPRO'
                          #  ])
    
    selected_name = st.selectbox("Select a name:",[" ",'Adani Enterprises','Adani Ports & SEZ','Apollo Hospitals','Asian Paints',
                                                    'Axis Bank','Bajaj Auto','Bajaj Finance',
                                                    'Bajaj Finserv','Bharat Petroleum','Bharti Airtel','Britannia Industries',
                                                    'Cipla','Coal India',"Divi's Laboratories","Dr. Reddy's Laboratories",
                                                    'Eicher Motors','Grasim Industries','HCLTech','HDFC Bank','HDFC Life',
                                        'Hero MotoCorp','Hindalco Industries','Hindustan Unilever',
                                        'ICICI Bank','IndusInd Bank','Infosys','ITC','JSW Steel','Kotak Mahindra Bank','Larsen & Toubro',
                                        'Mahindra & Mahindra','Maruti Suzuki','Nestlé India','NTPC','Oil & Natural Gas Corporation',
                                        'Power Grid','Reliance Industries','SBI Life Insurance Company',
                                        'State Bank of India','Sun Pharma','Tata Consultancy Services','Tata Motors',
                                        'Tata Consumer Products','Tata Steel','Tech Mahindra','Titan Company','UltraTech Cement','UPL','Wipro'
                                            ])
    

     # Define the folder path based on the selected name
    folder_path = os.path.join("E:\KaggleXProjects\Concalls", selected_name)
    #print(folder_path)
    # List PDF files in the selected folder
    pdf_files_names = list_pdfs_in_folder(folder_path)

    years = [extract_year_from_filename(os.path.basename(pdf_file)) for pdf_file in pdf_files_names]

    #st.write(years)
    #print(pdf_files)
     # Extract only the file names from the full paths
    #pdf_file_names = [os.path.basename(pdf_file) for pdf_file in pdf_files]

    if pdf_files_names:
        # Get unique years
        unique_years = list(set(years))

        # Create a dropdown for selecting the year
        selected_year = st.selectbox("Select a Year:", unique_years)

        # Filter PDFs based on the selected year
        selected_pdfs = [pdf_file for pdf_file, pdf_year in zip(pdf_files_names, years) if pdf_year == selected_year]

        if selected_pdfs:
            months = [get_month_from_filename(os.path.basename(pdf_file)) for pdf_file in selected_pdfs]

            # Get unique months
            unique_months = list(set(months))

            # Create a dropdown for selecting the month
            selected_month = st.selectbox("Select Month:", unique_months)
             # Filter PDFs based on the selected month
            selected_month_pdfs = [pdf_file for pdf_file, pdf_month in zip(selected_pdfs, months) if pdf_month == selected_month]

            return selected_month_pdfs

    return []




def main():
    # Create API client.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)

    # Get an access token from the credentials
    credentials.refresh(Request())
    access_token = credentials.token

    PROJECT_ID = "KaggleX-LLM"
# Set the headers for the API request
    #vertexai.init(project=PROJECT_ID, location="us-central1")
    load_dotenv()
    # Initialize session state
    st.set_page_config(page_title="Finpro.ai-FinGainInsights",
                       page_icon=":moneybag:")
    
   
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None



    st.header("Finpro.ai-Your financial Companion :moneybag:")
        
    #user_question = st.text_input("ASK a question?")
    #if user_question:
        #handle_userinput(user_question)

  
    #st.write(user_template.replace("{{MSG}}","Hello Finpro"),unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}","Hello Human"),unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Tokens")

        #pdf_docs=st.file_uploader(
            #"Chat with your Pdfs",accept_multiple_files=True)
        pdf_docs = folder_selector()
        #st.write(pdf_docs)
        
        if st.button("chat"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                vectorstore = get_vectorstore(text_chunks)

                #sequentialchain
                st.session_state.conversation= get_conversation_cahin(vectorstore)

                
    st.write(css,unsafe_allow_html=True)
   
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_query = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')
  


        if submit_button and user_query:
            with st.spinner('Generating response...'):
                handle_userinput(user_query)      
    

if __name__ == "__main__":
    main()
