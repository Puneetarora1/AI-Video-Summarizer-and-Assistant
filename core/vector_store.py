import os 
# from langchain_chroma import Chroma 
import chromadb
import uuid
# from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings(text : list):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(text, show_progress_bar=True)
    return embeddings
    
def build_vector_store(transcript : str)->chromadb:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description":"vector store collection of meeting transcript"}
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)
    
    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]
    
    texts = [doc.page_content for doc in docs] 
    
    embeddings = get_embeddings(texts)
    
    ids = []
    all_metadata = []
    documents_content = []
    embeddings_list = []
    
    for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
        doc_id = f"doc_{uuid.uuid4()}"
        ids.append(doc_id)
        
        metadata = dict(doc.metadata)
        metadata["content_length"] = len(doc.page_content)
        all_metadata.append(metadata)
        
        documents_content.append(doc.page_content)
        
        embeddings_list.append(embedding.tolist())
        
        vector_store.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )
        
    return vector_store
    

def get_retriever(vector_store : chromadb, query : str, k :int = 4, score_threshold : float=0.0)-> list:
    
    query_embeddings = get_embeddings([query])[0]
    
    # semantic search
    results = vector_store.query(
        query_embeddings=[query_embeddings.tolist()], 
        n_results=k
    )
    
    # cosine similarity
    retrieved_docs=[]
    
    if results["documents"] and results["documents"][0]:
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        
        for i, (doc_id, metadata, document, distance) in enumerate(zip(ids, metadatas, documents, distances)):
            similarity_score = 1-distance
            
            if similarity_score >= score_threshold:
                retrieved_docs.append({
                    "id" : doc_id,
                    "document" : document,
                    "metadata" : metadata,
                    "similarity_score" : similarity_score,
                    "rank" :i + 1
                })
                
        print(f"retrieved {len(retrieved_docs)} documents")
        
    else:
        print("no documents found")
        
    
    return retrieved_docs