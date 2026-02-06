from flask import Flask, request, jsonify, render_template
from src.helper import download_huggingface_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import requests  # for calling Ollama
from src.prompt import *
import base64  # for image encoding
import io
from PIL import Image

app = Flask(__name__)
load_dotenv()

# FAISS setup (local vector store)
print("Loading FAISS index...")
embeddings = download_huggingface_embeddings()
index_path = "faiss_index"

docsearch = FAISS.load_local(
    index_path,
    embeddings,
    allow_dangerous_deserialization=True  # Safe: loading our own index
)
print("✅ FAISS index loaded successfully")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

conversation_history = []

# Ollama-based medical assistant
def get_answer_from_ollama(query: str):
    global conversation_history

    # Step 1: Check if query is medically relevant
    relevance_prompt = medical_check_prompt.format(query=query)

    relevance_check = call_ollama(relevance_prompt).strip()
    if relevance_check.upper().startswith("NO"):
        return "⚠️ I specialize only in medical-related queries. Please ask me something about health or medicine."

    # Step 2: Retrieve top relevant docs from FAISS
    retrieved_docs = retriever.invoke(query)

    # Step 3: Join context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 4: Prepare conversation history (last 5 exchanges)
    history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['bot']}"
                              for h in conversation_history[-5:]])

    # Final prompt
    final_prompt = prompt.format(query=query, context=context, history_text=history_text)

    response = call_ollama(final_prompt)

    # Save history
    conversation_history.append({"user": query, "bot": response})

    return response


# Helper to call Ollama API
def call_ollama(prompt: str, model: str = "mistral"):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60  # 60 second timeout to prevent connection reset
        )
        data = res.json()
        return data.get("response", "⚠️ No response from Ollama.")
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The AI model took too long to respond. Please try a shorter question."
    except Exception as e:
        return f"⚠️ Error connecting to Ollama: {e}"


# Helper to call Ollama Vision API (LLaVA)
def call_ollama_vision(prompt: str, image_b64: str, model: str = "llava"):
    """
    Call Ollama's LLaVA model for vision + language tasks
    Args:
        prompt: Text instruction for the model
        image_b64: Base64 encoded image
        model: Vision model name (default: llava)
    Returns:
        Model response as string
    """
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            },
            timeout=120  # Vision models take longer
        )
        data = res.json()
        return data.get("response", "⚠️ No response from vision model.")
    except requests.exceptions.Timeout:
        return "⚠️ Vision model request timed out. Please try again with a smaller image."
    except Exception as e:
        return f"⚠️ Error calling vision model: {e}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_query = request.form["msg"]
    answer = get_answer_from_ollama(user_query)
    return jsonify({"answer": answer})


@app.route("/analyze-report", methods=["POST"])
def analyze_report():
    """
    Endpoint to analyze lab report images using LLaVA + FAISS RAG
    Expects: multipart/form-data with 'image' file
    Returns: JSON with extracted values and medical explanation
    """
    try:
        # Step 1: Get uploaded image
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Step 2: Convert image to base64
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Step 3: Use LLaVA to extract lab values
        extraction_prompt = """
        Analyze this medical lab report image carefully.
        Extract ALL test names, values, units, and reference ranges.
        Format your response as:
        Test Name: Value Units (Reference: range) - Status
        
        Example:
        Hemoglobin: 14.2 g/dL (Reference: 13.0-17.0) - Normal
        Glucose: 180 mg/dL (Reference: 70-100) - HIGH
        
        Be precise with numbers and units. Mark abnormal values.
        """
        
        print("📷 Analyzing image with LLaVA...")
        extracted_text = call_ollama_vision(extraction_prompt, image_b64)
        
        if extracted_text.startswith("⚠️"):
            return jsonify({"error": extracted_text}), 500
        
        print(f"✅ Extracted: {extracted_text[:200]}...")
        
        # Step 4: Parse extracted text to identify medical terms for FAISS query
        medical_terms_prompt = f"""
        From this lab report data:
        {extracted_text}
        
        List ONLY the abnormal/concerning medical terms that need explanation.
        One term per line. Focus on: high/low values, conditions, diseases mentioned.
        """
        
        medical_terms = call_ollama(medical_terms_prompt, model="mistral")
        
        # Step 5: Query FAISS with extracted medical terms
        print(f"🔍 Querying FAISS for: {medical_terms[:100]}...")
        
        # Split terms and query FAISS
        terms_list = [term.strip() for term in medical_terms.strip().split('\n') if term.strip()][:5]
        
        all_docs = []
        for term in terms_list:
            try:
                docs = retriever.invoke(term, k=2)
                all_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ Error retrieving for '{term}': {e}")
        
        # Get unique docs
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        context = "\n\n".join([doc.page_content for doc in unique_docs[:5]])
        
        # Step 6: Generate comprehensive explanation using Mistral + context
        explanation_prompt = f"""
        You are a medical assistant explaining lab results to a patient.
        
        Lab Report Data:
        {extracted_text}
        
        Medical Knowledge Context:
        {context}
        
        Provide a clear, structured explanation:
        1. Summary of all values
        2. Normal values (briefly)
        3. Abnormal values (detailed with implications)
        4. Recommendations
        
        Be empathetic, clear, and include medical disclaimer.
        """
        
        print("🤖 Generating explanation with Mistral...")
        explanation = call_ollama(explanation_prompt, model="mistral")
        
        # Step 7: Return results
        return jsonify({
            "success": True,
            "extracted_values": extracted_text,
            "explanation": explanation,
            "sources_count": len(unique_docs)
        })
    
    except Exception as e:
        print(f"❌ Error in analyze_report: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
