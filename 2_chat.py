from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv



load_dotenv()



### INDEXING..................................

# (1)  Document Ingestion: In document ingestion we will load the document, at the moment i am using the Youtube API to load the Youtube transcript

video_id = "LPZh9BOjkQs"  # only the ID, not the full URL
try:
    # if we don't care the best language, this return the best one
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plan text
    transcript = " ".join(snippet.text for snippet in fetched_transcript)

    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

    

'''
fetched_transcript → the object you got from .fetch(). It holds many small pieces inside it called snippets (each snippet = one small chunk of spoken text with timing info).
for snippet in fetched_transcript → loop through those snippets, one at a time. Each time through the loop, the current piece is temporarily called snippet.
snippet.text → for each snippet, grab only its .text part (just the spoken words, ignoring timestamp info like .start and .duration).
(snippet.text for snippet in fetched_transcript) → this whole thing together is a compact loop that produces one snippet.text after another, without needing a separate for block written out.
" ".join(...) → takes all those separate .text pieces and glues them into one single string, placing a space " " between each piece.
transcript = ... → the final combined string (the whole transcript as one block of text) is stored in the variable transcript.
In one sentence: "Go through every snippet in the transcript, grab just its text, and stick all those texts together into one long string, separated by spaces."

'''


# (2) Text chunking (text splitting)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,

)

# chunks = splitter.split_text(transcript)    # split_text() return plain string (No page_content)

chunks = splitter.create_documents([transcript])

'''

transcript → one big string containing the whole video's transcript text.

splitter.create_documents(...) → calls the create_documents method on your splitter object. It does two things internally:

Splits the text into smaller chunks (based on chunk_size=1000, chunk_overlap=200 that you set earlier)
Wraps each chunk into a Document object automatically (page_content = that chunk's text)

chunks = ... → stores the final result — a list of Document objects — in the variable chunks.

'''


'''
FAISS.from_documents() expects Document objects, not plain strings — because it needs .page_content (and optionally .metadata) on each item.

'''


# (3) Embedding Generation: converting chunks into a dense vector, for semantic search 

# Embedding model

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task='feature-extraction'
)

# (4) Vectoe store

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model

)

### Retriever.................................................................

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 4}
)


# Retriever input: query
# Retriever output: most similar relevent chunks (then after: most similar relevent chunks + query, to create a prompt)


### Augmentation:

# most similar relevent chunks + query, to create a prompt

prompt = PromptTemplate(
    template="""
    You are a helpful assistent.
    Answer ONLY from the provided transcript context.

    {context}
    Question: {question}
    """,

    input_variables = ['context', 'question']

)




### Generation........................................

 # Model


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)



# Building a Chian

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text
    


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),                  # the output of this chain is context_string
    'question': RunnablePassthrough()                                    # question

})


parser = StrOutputParser()


main_chain = parallel_chain | prompt | model | parser


result = main_chain.invoke('in between the time 1:13:00 and 1:14:00, they talk about fusion reactions etc, you were wrong?')

print(result)