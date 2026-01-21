from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
load_dotenv()


video_id="GV9AEEIeHrQ"
ytt_api = YouTubeTranscriptApi()
transcript=""
try:
    transcript_ls=ytt_api.fetch(video_id)
    for i in transcript_ls:
        transcript+=i.text+" "

except Exception as e:
    print("No caption found for this video.")

#Splitting the transcript into chunks of 1000 characters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks=text_splitter.create_documents([transcript])
print(f"Number of chunks: {len(chunks)}")

#Embedding the chunks using Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embedding_list=embeddings.embed_documents([chunk.page_content for chunk in chunks])


from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created with embedded chunks.")
print(vector_store.index_to_docstore_id)


#Retrieving similar chunks 
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
query = "what is mr strack is doing in the video?"
retrieved_docs=retriever.invoke(query)
#re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.9
)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
context="\n\n".join([i.page_content for i in retrieved_docs])
final_prompt = prompt.format(context=context, question=query)
response = model.invoke(final_prompt)
print("Response from the model:")
print(response)
