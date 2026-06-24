# 🏥 MediBot — AI Medical Assistant

> A privacy-first, locally-run AI chatbot that answers medical questions using Retrieval-Augmented Generation (RAG). All processing happens on your machine — no data leaves your device.

---

## 📸 What It Does

MediBot is a conversational health assistant powered by local large language models. Ask it anything medical — symptoms, conditions, treatments, medications — and it answers using a curated medical knowledge base, not just the LLM's general training data.

It also reads lab reports. Upload a photo of your blood test or pathology report and MediBot extracts every value, flags abnormals, and explains what they mean in plain language.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **RAG-Powered Answers** | Retrieves relevant chunks from medical PDFs before generating a response |
| 🔍 **Medical Filter** | Automatically rejects non-medical queries |
| 🗂️ **Conversation Memory** | Remembers your last 5 exchanges for natural follow-up questions |
| 🖼️ **Lab Report Analysis** | Upload an image → extracts values → explains results |
| 🎤 **Voice Input** | Speak your question directly into the browser |
| 🌐 **Hindi Support** | Full English ↔ Hindi translation for both input and output |
| 🔒 **100% Local** | Runs on Ollama — no API keys, no cloud, no data sharing |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (MediBot UI)                     │
│          Voice Input · Text Input · Image Upload             │
│               EN/HI Language Selection                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Flask Backend (app.py)                   │
│                                                              │
│   ┌──────────────────┐        ┌───────────────────────────┐ │
│   │  Medical Filter  │        │    Lab Report Analyzer    │ │
│   │  (Mistral LLM)   │        │  (LLaVA Vision + Mistral) │ │
│   └────────┬─────────┘        └─────────────┬─────────────┘ │
│            │                                │               │
│   ┌────────▼─────────┐        ┌─────────────▼─────────────┐ │
│   │  FAISS Retriever │        │      FAISS Retriever      │ │
│   │   (top-5 chunks) │        │  (per extracted med term) │ │
│   └────────┬─────────┘        └─────────────┬─────────────┘ │
│            │                                │               │
│   ┌────────▼─────────────────────────────────▼─────────────┐ │
│   │               Ollama (localhost:11434)                  │ │
│   │          Mistral 7B · LLaVA (vision model)             │ │
│   └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FAISS Vector Store (local)                   │
│           sentence-transformers/all-MiniLM-L6-v2            │
│                  384-dim · Cosine similarity                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 How the RAG Pipeline Works

Every chat message goes through a 5-step pipeline:

### Step 1 — Medical Relevance Filter
Before anything else, the query is sent to Mistral with a classification prompt:

```
Is this query related to medical or healthcare topics?
Answer only YES or NO.

Query: {your question}
```

If the answer is **NO**, MediBot immediately replies with a polite rejection. Non-medical questions never reach the retrieval or generation steps.

---

### Step 2 — Semantic Retrieval from FAISS
The validated query is converted into a **384-dimensional vector** using `sentence-transformers/all-MiniLM-L6-v2` and compared against all pre-indexed medical document chunks stored in FAISS. The **top 5 most semantically similar chunks** are retrieved.

```
Query: "What are the symptoms of diabetes?"
  ↓ embed (384-dim vector)
  ↓ cosine similarity search
  ↑ Top 5 chunks from medical PDFs about diabetes symptoms
```

---

### Step 3 — Context & History Assembly
The retrieved chunks are joined into a single context string. The last 5 turns of your conversation are formatted as a history block. Together they form a rich, grounded prompt:

```
You are a knowledgeable and concise medical assistant.

Conversation History:
User: What causes type 2 diabetes?
Assistant: Type 2 diabetes is caused by...

User Question: What are the early symptoms?

Retrieved Context:
[chunk 1] Early signs of diabetes include increased thirst...
[chunk 2] Frequent urination is one of the hallmark symptoms...
[chunk 3] ...
```

---

### Step 4 — Response Generation via Ollama
The assembled prompt is sent to **Mistral 7B** running locally through Ollama's REST API. The model generates a focused, paragraph-style answer — no bullet points, no hallucinated information outside the provided context.

```
POST http://localhost:11434/api/generate
{
  "model": "mistral",
  "prompt": "<assembled prompt>",
  "stream": false
}
```

---

### Step 5 — History Update
The exchange `{ user: query, bot: response }` is appended to an in-memory conversation history. The next message will include this turn as context, enabling natural multi-turn conversations like:

> *"What is hypertension?"* → answer  
> *"How do I manage it?"* → answer (aware of hypertension context)  
> *"Are there any foods I should avoid?"* → answer (still aware of full thread)

---

## 🖼️ Lab Report Analysis Pipeline

Upload any lab report image (JPG, PNG, etc.) through the 📎 button in the UI.

```
Image Upload
    │
    ▼
LLaVA (Vision LLM)
    Extracts: Test Name · Value · Unit · Reference Range · Status
    ↓
Mistral identifies abnormal/concerning terms
    ↓
FAISS retrieval for each term (up to 5 terms, top-2 docs each)
    ↓
Mistral generates patient-friendly explanation:
    1. Summary of all values
    2. Normal values (brief)
    3. Abnormal values (detailed with implications)
    4. Recommendations
```

The final response includes the raw extracted values, a plain-language explanation, and the number of medical sources referenced.

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python + Flask | Web server and API routing |
| **LLM (Text)** | Mistral 7B via Ollama | Answer generation, query classification |
| **LLM (Vision)** | LLaVA via Ollama | Lab report image understanding |
| **Vector Store** | FAISS (local) | Fast semantic document retrieval |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim text vectorization |
| **PDF Processing** | LangChain + PyPDF | Document loading and chunking |
| **Frontend** | Vanilla HTML/CSS/JS | Chat UI |
| **Voice** | Web Speech API | Browser-native speech recognition |
| **Translation** | MyMemory API | EN ↔ HI translation |

**Embedding Details:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: `384`
- Chunk size: `500` characters, `20` character overlap
- Similarity metric: Cosine
- Retrieval: Top-`k=5`

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Required Ollama Models

```bash
# Text model (required)
ollama pull mistral

# Vision model (required for lab report analysis)
ollama pull llava
```

### 5. Add Medical Documents

Place your medical PDF files in the `Data/` directory. These are the documents MediBot will learn from.

### 6. Build the FAISS Index

> ⚠️ **Skip this step** if the `faiss_index/` folder already exists in the project — the index is already built.

```bash
python store_index.py
```

This processes all PDFs, generates embeddings, and saves the FAISS index locally. Run it only when you add new documents.

### 7. Start Ollama

```bash
ollama serve
```

Keep this terminal open.

### 8. Run the Application

```bash
python app.py
```

Open your browser and visit: **http://localhost:5000**

---

## 📁 Project Structure

```
Medical-Chatbot/
│
├── app.py                  # Main Flask app — routes, RAG pipeline, Ollama calls
│
├── src/
│   ├── helper.py           # PDF loader, text splitter, embedding model loader
│   └── prompt.py           # Prompt templates (medical filter + main RAG prompt)
│
├── store_index.py          # One-time script: builds and saves FAISS index
│
├── Data/                   # Place your medical PDF documents here
│
├── faiss_index/            # Auto-generated FAISS vector store (don't delete)
│   ├── index.faiss
│   └── index.pkl
│
├── templates/
│   └── index.html          # Chat UI (voice, translation, image upload)
│
├── static/
│   └── styling.css         # UI styles and animations
│
└── requirements.txt        # Python dependencies
```

---

## 🖥️ UI Overview

The web interface (`MediBot`) is built as a single-page chat application:

- **Header** — App name with live "Online" status indicator
- **Chat area** — Scrollable message thread with user/bot avatars and timestamps
- **Typing indicator** — Animated dots shown while MediBot generates a response
- **Input bar** — Language selector · Image upload · Text field · Voice button · Send button
- **Animated background** — Floating medical cross symbols and pulsing circles

**Input modes:**
| Mode | How |
|------|-----|
| Text | Type and press Enter or click Send |
| Voice | Click the 🎤 button and speak |
| Image | Click the 🖼️ button and select a lab report photo |

**Language mode (EN/HI):**
- Select **Hindi** from the dropdown
- Type or speak in Hindi — it's auto-translated to English before being sent to the backend
- The bot's English response is translated back to Hindi before display

---

## 🤖 Models Used

| Model | Role | Timeout |
|-------|------|---------|
| `mistral` | Medical filter, answer generation, lab result explanation | 60 seconds |
| `llava` | Lab report image reading and value extraction | 120 seconds |

Both models run locally through Ollama and require no internet connection after the initial pull.

---

## 💡 Example Conversations

**Text query:**
```
You:    What is hypertension?
MediBot: Hypertension, commonly known as high blood pressure, is a chronic
         condition where the force of blood against artery walls is consistently
         too high. It is defined as a reading of 130/80 mmHg or higher...
```

**Follow-up (conversation memory in action):**
```
You:    What foods should I avoid?
MediBot: For managing hypertension, it's important to limit sodium intake,
         processed foods, and excessive alcohol. The DASH diet...
```

**Non-medical query (filter in action):**
```
You:    How do I make pasta?
MediBot: ⚠️ I specialize only in medical-related queries.
         Please ask me something about health or medicine.
```

---

## ⚠️ Disclaimer

MediBot is intended for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.