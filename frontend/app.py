import streamlit as st
import os
import requests
import socket

# Force IPv4 DNS resolution to prevent slow Windows IPv6 lookup timeouts on external APIs
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return orig_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = patched_getaddrinfo

# Point to the FastAPI backend API
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Premium CSS Styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617);
    color: #f8fafc;
}

/* Chat container glassmorphism */
div[data-testid="stChatMessage"] {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 12px !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stChatMessage"]:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.1) !important;
}

/* User message unique accent */
div[data-testid="stChatMessage"][data-testid$="user"] {
    border-left: 4px solid #6366f1 !important;
}

/* Assistant message unique accent */
div[data-testid="stChatMessage"][data-testid$="assistant"] {
    border-left: 4px solid #10b981 !important;
}

/* Header style */
h1 {
    background: linear-gradient(to right, #6366f1, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

/* Input box styling */
div[data-testid="stChatInput"] {
    border-radius: 12px !important;
}

/* Chat input sticky container glassmorphic background */
div[data-testid="stBottom"] {
    background: transparent !important;
}

div[data-testid="stBottom"] > div {
    background: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 10px 20px 20px 20px !important;
}

/* Style input area itself */
textarea[data-testid="stChatInputTextArea"] {
    background-color: rgba(30, 41, 59, 0.5) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Form styling */
div[data-testid="stForm"] {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
</style>
"""

st.set_page_config(page_title="UG Prospectus Chatbot", page_icon="🎓", layout="centered")
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("<h1>🎓 UG Prospectus AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1em;'>Interact with the official University Undergraduate Prospectus. Get accurate answers instantly.</p>", unsafe_allow_html=True)
st.markdown("---")

# Session state setup
if "session_token" not in st.session_state:
    st.session_state.session_token = None
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.messages = []
    st.session_state.pending_query = None

# --- 1. Authentication View ---
if st.session_state.session_token is None:
    st.markdown("### Authentication Required")
    auth_mode = st.radio("Choose Action", ["Login", "Signup"], horizontal=True)
    
    with st.form("auth_form"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            if not username or not password:
                st.error("Username and password are required")
            else:
                if auth_mode == "Login":
                    try:
                        resp = requests.post(f"{API_URL}/auth/login", json={"username": username, "password": password})
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.session_token = data["token"]
                            st.session_state.role = data["role"]
                            st.session_state.username = data["username"]
                            st.session_state.messages = []
                            st.success(f"Welcome back, {username}!")
                            if hasattr(st, "rerun"):
                                st.rerun()
                            else:
                                st.experimental_rerun()
                        else:
                            st.error(f"Login failed: {resp.json().get('detail', 'Unknown username or password')}")
                    except Exception as e:
                        st.error(f"API Connection Error: {e}")
                else:
                    try:
                        resp = requests.post(f"{API_URL}/auth/signup", json={"username": username, "password": password})
                        if resp.status_code == 200:
                            st.success("Signup successful! Please select 'Login' and log in.")
                        else:
                            st.error(f"Signup failed: {resp.json().get('detail', 'Username already exists or invalid')}")
                    except Exception as e:
                        st.error(f"API Connection Error: {e}")
    st.stop()

# --- 2. Main Interface ---

# Sidebar User Panel & Navigation
with st.sidebar:
    st.markdown(f"### Signed in as **{st.session_state.username}**")
    st.markdown(f"Role: `{st.session_state.role}`")
    st.markdown("---")
    
    if st.session_state.role == "ADMIN":
        st.markdown("### Navigation")
        app_mode = st.radio(
            "Go to",
            ["💬 Chatbot Interface", "📤 Ingest New Prospectus", "🔑 Create New Admin"],
            index=0
        )
        st.markdown("---")
    else:
        app_mode = "💬 Chatbot Interface"
        
    if st.button("Logout", use_container_width=True):
        st.session_state.session_token = None
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# Render View Based on Navigation selection
if app_mode == "💬 Chatbot Interface":
    # Display conversation history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "Seat Distribution Matrix" in message["content"]:
                try:
                    resp = requests.get(f"{API_URL}/seat_distribution.pdf", timeout=10)
                    if resp.status_code == 200:
                        st.download_button(
                            label="📥 Download Seat Distribution PDF",
                            data=resp.content,
                            file_name="seat_distribution.pdf",
                            mime="application/pdf",
                            key=f"download_{i}"
                        )
                except Exception as e:
                    print(f"Error fetching PDF from backend: {e}")

    # Handle pending assistant response (streaming)
    if st.session_state.get("pending_query"):
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        
        with st.chat_message("assistant"):
            try:
                headers = {"Authorization": f"Bearer {st.session_state.session_token}"}
                # History is everything before the last user query (which was already appended)
                history_payload = st.session_state.messages[:-1]
                resp = requests.post(
                    f"{API_URL}/user/query", 
                    json={"query": query, "history": history_payload}, 
                    headers=headers,
                    stream=True
                )
                if resp.status_code == 200:
                    resp.encoding = resp.encoding or "utf-8"

                    def get_stream():
                        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                yield chunk
                                
                    answer = st.write_stream(get_stream())
                else:
                    try:
                        detail = resp.json().get('detail', 'Unknown server error')
                    except Exception:
                        detail = resp.text or 'Unknown server error'
                    answer = f"⚠️ **Error:** {detail}"
                    st.markdown(answer)
            except Exception as e:
                answer = f"⚠️ **Connection Error:** {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # Chat input box (At root page level, so Streamlit pins it to the bottom of the viewport)
    question = st.chat_input("Ask a question about courses, eligibility, or admissions...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending_query = question
        st.rerun()

    # JS helper to auto-scroll chat to the bottom on rerun / query completion
    st.iframe(
        """
        <script>
            setTimeout(function() {
                var scrollContainer = window.parent.document.querySelector('.main');
                if (scrollContainer) {
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                }
            }, 100);
        </script>
        """,
        height=1,
        width=1
    )

elif app_mode == "📤 Ingest New Prospectus":
    st.markdown("### Upload & Process New Prospectus")
    st.write("Upload a new PDF prospectus. You can specify which 0-based pages contain seat distribution matrices to split them from RAG parsing.")
    
    uploaded_file = st.file_uploader("Select Prospectus PDF", type=["pdf"])
    ex_pages = st.text_input("Excluded Seat Distribution Pages (comma-separated, e.g. 79,80,81)", "79,80,81")
    
    if st.button("Upload & Start Processing", use_container_width=True):
        if not uploaded_file:
            st.error("Please select a PDF file first.")
        else:
            with st.spinner("Uploading file and initiating processing pipeline..."):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.session_token}"}
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"excluded_pages": ex_pages}
                    
                    resp = requests.post(
                        f"{API_URL}/admin/upload-prospectus", 
                        files=files, 
                        data=data, 
                        headers=headers
                    )
                    
                    if resp.status_code == 200:
                        res = resp.json()
                        st.success(res["message"])
                        st.info(f"Page Exclusions Applied: {res['excluded_pages_applied']}")
                    else:
                        st.error(f"Failed to process upload: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

elif app_mode == "🔑 Create New Admin":
    st.markdown("### Register a New Administrator")
    st.write("Create a new user profile with full administrative permissions.")
    
    with st.form("create_admin_form"):
        new_admin_username = st.text_input("New Admin Username").strip()
        new_admin_password = st.text_input("New Admin Password", type="password")
        create_submitted = st.form_submit_button("Register Admin User")
        
        if create_submitted:
            if not new_admin_username or not new_admin_password:
                st.error("Both username and password are required")
            else:
                with st.spinner("Registering admin..."):
                    try:
                        headers = {"Authorization": f"Bearer {st.session_state.session_token}"}
                        resp = requests.post(
                            f"{API_URL}/admin/create-admin",
                            json={"username": new_admin_username, "password": new_admin_password},
                            headers=headers
                        )
                        if resp.status_code == 200:
                            st.success(resp.json()["message"])
                        else:
                            st.error(f"Failed to register admin: {resp.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
