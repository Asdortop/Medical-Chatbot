from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from src.prompt import *

app = Flask(__name__)
load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

embeddings = download_huggingface_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings,
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k":5})

conversation_history = []

def get_answer_from_gemini(query: str):
    global conversation_history
    # Step 1: Check if query is medically relevant
    relevance_prompt = medical_check_prompt.format(query=query)

    relevance_check = model.generate_content(relevance_prompt).text.strip()

    if relevance_check.upper().startswith("NO"):
        return "⚠️ I specialize only in medical-related queries. Please ask me something about health or medicine."

    # Step 2: Retrieve top relevant docs from Pinecone
    retrieved_docs = retriever.invoke(query)

    # Step 3: Join context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 4: Prepare conversation history (last 5 exchanges)
    history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['bot']}" 
                              for h in conversation_history[-5:]])
    
    final_prompt = prompt.format(query=query, context=context, history_text=history_text)
    response = model.generate_content(final_prompt).text

    conversation_history.append({"user": query, "bot": response})

    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods = ["GET","POST"])
def chat():
    user_query = request.form["msg"]
    answer = get_answer_from_gemini(user_query)
    return jsonify({"answer": answer})
    

if __name__ == "__main__":
    app.run(debug=True)

