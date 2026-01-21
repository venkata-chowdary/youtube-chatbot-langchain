from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

video_id="UtcPMi3HU-M"
api=YouTubeTranscriptApi()
transcript=""
try:
    api_res=api.fetch(video_id)
    for i in api_res:
        transcript+=i.text+" "
except Exception as e:
    print("No caption found for this video.")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks=text_splitter.create_documents([transcript])
print(f"Number of chunks: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# chunk_embeddings = embeddings.embed_documents([doc.page_content for doc in chunks])
# print("Embeddings for chunks created.")

vs=FAISS.from_documents(chunks, embeddings)
print("Vector store created with embedded chunks.")


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


parser = StrOutputParser()
query="what is the guy doing in the video?"

retriever=vs.as_retriever(search_type="similarity", search_kwargs={"k":4})

retriever_runnable =RunnableLambda(lambda x: {
    "question":x['question'],
    "context":"\n\n".join(i.page_content for i in retriever.invoke(x['question']))
    })

chain=RunnableSequence(retriever_runnable | prompt | model | parser)
result=chain.invoke({"question":query})
print("Answer:", result)