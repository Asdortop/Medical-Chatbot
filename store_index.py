from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from langchain_community.vectorstores import FAISS
import os

print("Loading and processing medical documents...")
extracted_data = load_pdf_file(data="Data/")
print(f"✅ Loaded {len(extracted_data)} pages from PDFs")

print("Splitting documents into chunks...")
text_chunks = text_split(extracted_data)
print(f"✅ Created {len(text_chunks)} text chunks")

print("Loading embedding model...")
embeddings = download_huggingface_embeddings()
print("✅ Embedding model loaded")

print("\nCreating FAISS index...")
docsearch = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)
print("✅ FAISS index created in memory")

# Save to disk
index_path = "faiss_index"
print(f"\nSaving FAISS index to '{index_path}/'...")
docsearch.save_local(index_path)
print(f"✅ FAISS index saved successfully!")
print(f"   - Location: {os.path.abspath(index_path)}/")
print(f"   - Total chunks indexed: {len(text_chunks)}")
print(f"   - Embedding dimensions: 384")
print("\n🎉 Migration complete! You can now run app.py")

