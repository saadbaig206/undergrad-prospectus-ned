import os
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
PINECONE_INDEX_NAME = "rag-chatbot-index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# In production, host 'seat_distribution.pdf' on AWS S3 / Cloudinary and use that URL here
SEAT_DIST_FILE_LINK = "http://localhost:8000/seat_distribution.pdf" 

# Initialize Clients
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def is_seat_distribution_query(user_query: str) -> bool:
    """Uses Groq LLM to safely classify the user's intent."""
    prompt = f"""
You are a binary query classifier.

Your task is to determine whether the user's query is related to seat distribution or seat availability.

Respond with exactly:
- "YES" → if the query is about seats in any way.
- "NO" → otherwise.

Return "YES" if the query asks about any of the following:

• Total number of seats
• Seat distribution
• Seat matrix
• Seat allocation
• Number of seats in a specific department, program, discipline, or field
• Intake capacity
• Available seats
• Reserved seats
• Category-wise seats (General, OBC, SC, ST, EWS, Self-Finance, Open Merit, Quota, etc.)
• Merit-wise seat distribution
• Campus-wise seat distribution
• Shift-wise seats (Morning/Evening)
• Gender-wise seat allocation
• Quota-wise seats
• Department-wise seats
• Faculty-wise seats
• Any question asking "How many seats..." for any program, course, campus, category, department, or specialization.

Examples that should return YES:
- How many seats are there in Computer Science?
- CS has how many seats?
- Seat matrix of Engineering.
- Show seat distribution for BS AI.
- Seats for Self Finance?
- Reserved seats for Sindh Rural?
- Intake of Software Engineering.
- How many students are admitted in Electrical Engineering?
- Department-wise seat allocation.
- Campus-wise seat distribution.
- Total seats in AI.
- Merit seats for Data Science.

Return "NO" if the query is about:
• Eligibility
• Admissions
• Registration
• Fees
• Scholarships
• Merit calculation
• Admission dates
• Deadlines
• Documents
• Syllabus
• Courses
• Curriculum
• Faculty
• Hostel
• Transport
• Exam schedule
• General university information
• Any topic unrelated to seat counts or seat allocation.

Query:
"{user_query}"

Response (YES/NO):
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content.strip().upper()
    return "YES" in answer

def get_rag_response(user_query: str) -> str:
    """Retrieves relevant chunks from Pinecone and asks Groq to answer."""
    # 1. Embed query locally
    query_vector = embed_model.encode(user_query).tolist()
    
    # 2. Query Pinecone
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    # 3. Build Context
    context_chunks = [match['metadata']['text'] for match in results['matches']]
    context = "\n---\n".join(context_chunks)
    
    # 4. Generate Answer via Groq
    system_prompt = """You are a highly capable academic assistant. Your goal is to answer the user's question as thoroughly and helpful as possible using the retrieved context from the official Undergraduate Prospectus 2025.

Instructions:
1. Provide a direct, detailed, and clear answer.
2. Structure your response with bullet points and bold text for readability.
3. If the context does not contain the exact specific detail (like a date or specific value), do NOT say 'I don't know' or 'I don't have the information'. Instead, explain what the document states generally about the topic, cite the relevant section name/number from the context, and guide the user on how they can find the official update (e.g., checking the university's official website or admission office as mentioned in section X).
4. Always maintain a professional, encouraging, and authoritative tone. Avoid negative phrasing like 'I cannot answer' or 'this is not mentioned'."""
    user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", # Using 70b for higher quality reasoning
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def chat_router(user_query: str):
    """Routes the query dynamically based on content type."""
    print(f"\nUser: {user_query}")
    
    if is_seat_distribution_query(user_query):
        print(f"Bot: For the complete and accurate Seat Distribution Matrix, please refer to the official document: [Seat Distribution PDF]{SEAT_DIST_FILE_LINK}.")
    else:
        answer = get_rag_response(user_query)
        print(f"Bot: {answer}")

# --- Test Interface ---
if __name__ == "__main__":
    # Test cases
    chat_router("What is the eligibility criteria for the course?")
    chat_router("Can you show me the seat distribution chart for OBC and General categories?")
    chat_router("When is the last date to submit the application form?")