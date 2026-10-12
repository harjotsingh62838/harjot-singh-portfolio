import os
import sys
import time
import docx
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import pyttsx3
import whisper
import chromadb
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()
# STEP 1: LOAD & PARSE THE COMPANY KNOWLEDGE BASE (.docx)
def load_honey_knowledge_base(file_path="Honey_Company_Knowledge_Base.docx"):
    """
    Extracts all clean text paragraphs from the Honey Word document.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: Knowledge Base document '{file_path}' not found!")

    print(f" STEP 1: Loading knowledge base from '{file_path}'...")
    doc = docx.Document(file_path)

    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:  # Ignore empty lines
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    print(f" Successfully loaded document! Total character length: {len(full_text)} characters across {len(paragraphs)} paragraphs.")
    return full_text

# STEP 2: SPLIT TEXT INTO SMALLER CHUNKS FOR VECTOR RETRIEVAL
def chunk_knowledge_base(text):
    """
    Splits the full knowledge base text into overlapping chunks using RecursiveCharacterTextSplitter.
    Smaller chunks (400 chars) work best for fast, accurate semantic search in voice RAG.
    """
    print("\n STEP 2: Splitting knowledge base into manageable text chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f" Created {len(chunks)} text chunks.")
    return chunks

# STEP 3: INITIALIZE EMBEDDINGS & CHROMADB VECTOR DATABASE
def initialize_vector_store(chunks, db_path="./chroma_honey_db"):
    """
    Generates 384-dimensional vector embeddings for each chunk using SentenceTransformers
    and stores them inside ChromaDB persistent collection.
    """
    print("\n STEP 3: Loading local embedding model ('all-MiniLM-L6-v2')...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Initializing ChromaDB vector store at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    # Get or create collection
    collection = client.get_or_create_collection(name="honey_receptionist_kb")

    # If collection is empty, index all chunks
    if collection.count() == 0:
        print(f" Embedding and indexing {len(chunks)} chunks into ChromaDB...")
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = embedding_model.encode(chunks).tolist()
        metadatas = [{"source": "Honey_Company_Knowledge_Base.docx", "chunk_id": i} for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f" indexed{collection.count()}chunks into chromaDB vector database successfully!")
    else:
        print(f"found existing vector collection with {collection.count()}indexed chunks.")
    return collection, embedding_model

# STEP 4: RAG SEMANTIC SEARCH RETRIEVER
def retrieve_relevant_context(user_query, collection, embedding_model, top_k=3):
    """
    Converts user's query into an embedding vector, performs cosine similarity search
    in ChromaDB, and returns top K matching chunks as context.
    """
    # Convert query text into vector
    query_vector = embedding_model.encode([user_query]).tolist()
    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )

    retrieved_documents = results.get("documents", [[]])[0]
    combined_context = "\n---\n".join(retrieved_documents)
    return combined_context

# STEP 5: SPEECH-TO-TEXT (STT) ENGINE USING WHISPER
class VoiceRecorderSTT:
    def __init__(self, model_size="tiny.en"):
        print(f"\n STEP 5: Loading OpenAI Whisper model ('{model_size}')...")
        self.stt_model = whisper.load_model(model_size)
        print(" Whisper Speech-to-Text engine ready!")

    def record_audio(self, filename="user_input.wav", duration=5, sample_rate=16000):
        """Records microphone input for 'duration' seconds and saves as WAV file."""
        print(f"\n [LISTENING...] Speak into microphone for {duration} seconds...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording finishes
        print("\n [STOPPED LISTENING] Processing audio file...")
        # Convert float32 array to 16-bit PCM WAV
        audio_int16 = (recording * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
        return filename

    def transcribe_audio(self, audio_file="user_input.wav"):
        """Transcribes WAV audio file into text using Whisper."""
        try:
            sr, y = wavfile.read(audio_file)
            audio_data = y.astype(np.float32) / 32768.0

            # Transcribe audio array directly
            result = self.stt_model.transcribe(audio_data, fp16=False)
            transcription = result.get("text", "").strip()
            return transcription
        except Exception as e:
            print(f" Speech Transcription Error: {e}")
            return ""

# STEP 6: TEXT-TO-SPEECH (TTS) ENGINE FOR AI VOICE RESPONSE
def speak_text(text):
    """
    Converts LLM response text into spoken voice audio using pyttsx3 offline engine.
    """
    print(f"\n [AI VOICE RECEPTIONIST]: {text}")
    try:
        engine = pyttsx3.init()
        # Adjust speaking rate for warm human clarity (160 words/min)
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Voice Output Error (fallback to printed text only): {e}")

def generate_receptionist_response(user_query, context, client):
    system_prompt = """You are the official AI Voice Receptionist for Honey, a full-service technology and digital solutions company.

GUIDELINES FOR YOUR RESPONSE:
1. Tone: Warm, professional, polite, and concise (suitable for voice speaking out loud).
2. Grounding: Answer ONLY based on the provided retrieved context from Honey Knowledge Base. Do NOT make up information.
3. Brevity: Keep responses short and conversational (2-4 sentences) so it sounds natural when spoken over voice.
4. Escalation: If asked for formal pricing quotes, payment disputes, or custom contracts, offer to collect their contact details (name, email, phone) so the human team can assist them.
5. If the answer is not in the context, politely say: "I don't have that specific detail right now, but I can connect you with a Honey representative. May I have your name and email address?"
"""

    user_prompt = f"""RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

USER QUESTION:
{user_query}

RECEPTIONIST VOICE RESPONSE:"""

    # List candidate models to try in order of availability
    candidate_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192"
    ]

    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Try next model if model not found or rate limited
            continue

    return "I am sorry, I am currently unable to process your request. Please try again shortly."

# STEP 8: MAIN INTERACTIVE RECEPTIONIST VOICE LOOP
def main():
    print("\n" + "=" * 70)
    print("🤖 HONEY RAG-BASED AI VOICE RECEPTIONIST")
    print("=" * 70)

    # Verify Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file! Please add it to run LLM generation.")
        return

    # INITIALIZE GROQ CLIENT
    groq_client = Groq(api_key=api_key)

    # 1. Load Knowledge Base
    kb_text = load_honey_knowledge_base("Honey_Company_Knowledge_Base.docx")

    # 2. Split into Chunks
    chunks = chunk_knowledge_base(kb_text)

    # 3. Initialize ChromaDB Vector DB
    collection, embedding_model = initialize_vector_store(chunks)

    # 4. Initialize Whisper STT Engine
    stt_engine = VoiceRecorderSTT(model_size="tiny.en")

    # Greet Visitor
    initial_greeting = "Hello and welcome to Honey! I am your AI Receptionist. How can I assist you today?"
    speak_text(initial_greeting)

    print("\n" + "-" * 70)
    print("💡 INPUT MODE CHOICES:")
    print("1. Voice Mode (Speak into Microphone)")
    print("2. Text Mode (Type your question)")
    print("Type 'exit' or 'quit' at any time to end the call.")
    print("-" * 70)

    while True:
        mode = input("\n👉 Choose Input Mode (1 = Voice, 2 = Text, 'exit' to quit): ").strip().lower()

        if mode in ['exit', 'quit']:
            farewell = "Thank you for contacting Honey. Have a wonderful day!"
            speak_text(farewell)
            break

        user_query = ""

    # Voice Input Mode
        if mode == '1':
            try:
                duration_str = input("⏱️ Set recording duration in seconds (default 5): ").strip()
                duration = int(duration_str) if duration_str.isdigit() else 5
                audio_file = stt_engine.record_audio(duration=duration)
                print("⌛ Transcribing voice using OpenAI Whisper...")
                user_query = stt_engine.transcribe_audio(audio_file)
                print(f"🗨️ You said: \"{user_query}\"")
            except Exception as e:
                print(f"⚠️ Microphone or Voice error: {e}")
                continue  # ONLY continue when an error happens

        # Text Input Mode
        elif mode == '2':
            user_query = input("💬 Type your question for Honey Receptionist: ").strip()
        else:
            print("⚠️ Invalid choice. Please enter 1 for Voice, 2 for Text, or 'exit'.")
            continue

        if not user_query:
            print("⚠️ No input detected. Please try asking a question again.")
            continue

        if user_query.lower() in ['exit', 'quit']:
            farewell = "Thank you for contacting Honey. Have a wonderful day!"
            speak_text(farewell)
            break

        # 5. Retrieve Grounded Context from Vector Store
        print("\n🔍 Searching Honey Knowledge Base...")
        context = retrieve_relevant_context(user_query, collection, embedding_model, top_k=3)

        # 6. Generate Response via Groq LLM
        print("🧠 Generating grounded response from Groq Llama 3.3 70B...")
        ai_response = generate_receptionist_response(user_query, context, groq_client)

        # 7. Speak out Response
        speak_text(ai_response)


if __name__ == "__main__":
    main()