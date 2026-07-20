import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Point to backend API
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function App() {
  // Session State
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [role, setRole] = useState<string | null>(localStorage.getItem('role'));
  const [username, setUsername] = useState<string | null>(localStorage.getItem('username'));

  // Auth Forms State
  const [isLogin, setIsLogin] = useState(true);
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');

  // UI Panels State
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<'chat' | 'ingest' | 'register-admin'>('chat');

  // Chat State
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [usePgKnowledge, setUsePgKnowledge] = useState(false);

  // Ingestion State
  const [academicLevel, setAcademicLevel] = useState<'undergraduate' | 'postgraduate'>('undergraduate');
  const [extractSeats, setExtractSeats] = useState(false);
  const [excludedPages, setExcludedPages] = useState('79,80,81');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [ingestStatus, setIngestStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle');
  const [ingestMsg, setIngestMsg] = useState('');

  // Admin Registration State
  const [newAdminUser, setNewAdminUser] = useState('');
  const [newAdminPass, setNewAdminPass] = useState('');
  const [adminRegError, setAdminRegError] = useState('');
  const [adminRegSuccess, setAdminRegSuccess] = useState('');

  // Auto-scroll chat to bottom
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  // Ingestion status checker
  const checkIngestionStatus = async (level: 'undergraduate' | 'postgraduate') => {
    if (!token) return;
    try {
      const res = await fetch(`${API_URL}/admin/ingestion-status?academic_level=${level}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'processing') {
          setIngestStatus('processing');
          setIngestMsg('Indexing & splitting PDF chunks in background...');
        } else if (data.status === 'completed') {
          setIngestStatus('success');
          setIngestMsg('Ingestion and indexing completed successfully!');
        } else if (data.status === 'failed') {
          setIngestStatus('error');
          setIngestMsg(`Ingestion failed: ${data.error || 'Unknown error'}`);
        } else {
          setIngestStatus('idle');
          setIngestMsg('');
        }
      }
    } catch (err) {
      console.error('Failed to fetch ingestion status:', err);
    }
  };

  // Check status on tab or level change
  useEffect(() => {
    if (activeTab === 'ingest' && token) {
      checkIngestionStatus(academicLevel);
    }
  }, [activeTab, academicLevel, token]);

  // Poll status while processing
  useEffect(() => {
    let intervalId: any;
    if (ingestStatus === 'processing' && token) {
      intervalId = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/admin/ingestion-status?academic_level=${academicLevel}`, {
            headers: { 'Authorization': `Bearer ${token}` }
          });
          if (res.ok) {
            const data = await res.json();
            if (data.status === 'completed') {
              setIngestStatus('success');
              setIngestMsg('Ingestion and indexing completed successfully!');
              clearInterval(intervalId);
            } else if (data.status === 'failed') {
              setIngestStatus('error');
              setIngestMsg(`Ingestion failed: ${data.error || 'Unknown error'}`);
              clearInterval(intervalId);
            }
          }
        } catch (err) {
          console.error('Error polling ingestion status:', err);
        }
      }, 3000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [ingestStatus, academicLevel, token]);

  // Auth Handlers
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError('');
    setAuthSuccess('');

    if (!authUsername.trim() || !authPassword) {
      setAuthError('All fields are required.');
      return;
    }

    const endpoint = isLogin ? '/auth/login' : '/auth/signup';
    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: authUsername.trim(), password: authPassword }),
      });

      if (!res.ok) {
        const errData = await res.json();
        setAuthError(errData.detail || 'Authentication failed.');
        return;
      }

      if (isLogin) {
        const data = await res.json();
        localStorage.setItem('token', data.token);
        localStorage.setItem('role', data.role);
        localStorage.setItem('username', data.username);
        setToken(data.token);
        setRole(data.role);
        setUsername(data.username);
        setMessages([]);
      } else {
        setAuthSuccess('Signup successful! Switch to Login to sign in.');
        setAuthUsername('');
        setAuthPassword('');
      }
    } catch (err) {
      setAuthError('Connection error: backend is unreachable.');
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    setToken(null);
    setRole(null);
    setUsername(null);
    setMessages([]);
    setActiveTab('chat');
  };

  // Trigger streaming message
  const triggerQuery = async (queryText: string) => {
    if (!queryText.trim() || isGenerating || !token) return;

    setIsGenerating(true);
    const newMessages = [...messages, { role: 'user' as const, content: queryText }];
    setMessages(newMessages);

    // Placeholder message for live streaming
    setMessages((prev) => [...prev, { role: 'assistant' as const, content: '' }]);

    try {
      const historyPayload = newMessages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetch(`${API_URL}/user/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ query: queryText, history: historyPayload, use_pg_knowledge: usePgKnowledge }),
      });

      if (!res.ok) {
        const errData = await res.json();
        const errText = errData.detail || 'An error occurred during query generation.';
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: `⚠️ **Error:** ${errText}` };
          return updated;
        });
        setIsGenerating(false);
        return;
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      if (!reader) throw new Error('Could not read stream body.');

      let assistantAnswer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        assistantAnswer += chunk;

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: assistantAnswer };
          return updated;
        });
      }
    } catch (err: any) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: 'assistant', content: `⚠️ **Connection Error:** ${err.message}` };
        return updated;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isGenerating) return;
    const text = chatInput.trim();
    setChatInput('');
    triggerQuery(text);
  };

  // Copy to clipboard helper
  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // PDF Matrix Downloader
  const downloadSeatMatrix = async () => {
    try {
      const res = await fetch(`${API_URL}/seat_distribution.pdf`);
      if (!res.ok) throw new Error('PDF not found.');
      const blob = await res.blob();
      const fileUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = fileUrl;
      link.setAttribute('download', 'seat_distribution.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      alert('Error downloading Seat Distribution PDF.');
    }
  };

  // Ingestion File Uploader
  const handleIngestSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile || !token) return;

    setIngestStatus('uploading');
    setIngestMsg('');

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('academic_level', academicLevel);
    formData.append('excluded_pages', (academicLevel === 'undergraduate' && extractSeats) ? excludedPages : '');

    try {
      const res = await fetch(`${API_URL}/admin/upload-prospectus`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        setIngestStatus('error');
        setIngestMsg(errData.detail || 'Ingestion upload failed.');
        return;
      }

      setIngestStatus('processing');
      setIngestMsg('Processing parallel extraction and indexing in backend...');
      setUploadFile(null);

    } catch (err) {
      setIngestStatus('error');
      setIngestMsg('Upload connection timed out.');
    }
  };

  // Register New Admin
  const handleRegisterAdmin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAdminRegError('');
    setAdminRegSuccess('');

    if (!newAdminUser.trim() || !newAdminPass) {
      setAdminRegError('All fields are required.');
      return;
    }

    try {
      const res = await fetch(`${API_URL}/admin/create-admin`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ username: newAdminUser.trim(), password: newAdminPass }),
      });

      if (!res.ok) {
        const errData = await res.json();
        setAdminRegError(errData.detail || 'Failed to create administrator.');
        return;
      }

      setAdminRegSuccess('Administrator registered successfully!');
      setNewAdminUser('');
      setNewAdminPass('');
    } catch (err) {
      setAdminRegError('Connection failure.');
    }
  };


  // --- Auth View Render ---
  if (!token) {
    return (
      <div className="auth-panel-wrapper">
        <div className="auth-panel-card">
          <img src="/image.png" alt="Prospectus AI Logo" style={{ height: '80px', width: 'auto', marginBottom: '8px', borderRadius: '8px' }} onError={(e) => { e.currentTarget.style.display = 'none'; }} />
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '24px' }}>
            Official Academic RAG Assistant
          </p>

          <div className="auth-tab-row">
            <button
              onClick={() => { setIsLogin(true); setAuthError(''); setAuthSuccess(''); }}
              className={`auth-tab-btn ${isLogin ? 'active' : ''}`}
            >
              Log In
            </button>
            <button
              onClick={() => { setIsLogin(false); setAuthError(''); setAuthSuccess(''); }}
              className={`auth-tab-btn ${!isLogin ? 'active' : ''}`}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'left' }}>
            <div className="form-group">
              <label>Username</label>
              <input
                type="text"
                value={authUsername}
                onChange={(e) => setAuthUsername(e.target.value)}
                placeholder="Enter your username"
                required
              />
            </div>
            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                placeholder="Enter your password"
                required
              />
            </div>

            {authError && <div style={{ color: '#ef4444', fontSize: '0.9rem', fontWeight: 600 }}>⚠️ {authError}</div>}
            {authSuccess && <div style={{ color: '#10b981', fontSize: '0.9rem', fontWeight: 600 }}>✅ {authSuccess}</div>}

            <button type="submit" className="btn-primary" style={{ marginTop: '10px' }}>
              {isLogin ? 'Log In' : 'Sign Up'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // --- Main ChatGPT Dashboard Render ---
  return (
    <div className="app-container">
      {/* Sidebar Panel */}
      <div className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header" style={{ justifyContent: 'center', padding: '16px' }}>
          <img src="/image.png" alt="Prospectus AI Logo" className="sidebar-logo" style={{ height: 'auto', width: '60px', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }} onError={(e) => { e.currentTarget.style.display = 'none'; }} />
        </div>

        <div className="sidebar-content">
          <button
            onClick={() => setActiveTab('chat')}
            className={`sidebar-tab ${activeTab === 'chat' ? 'active' : ''}`}
          >
            💬 Chat Workspace
          </button>
          
          {role === 'ADMIN' && (
            <>
              <button
                onClick={() => setActiveTab('ingest')}
                className={`sidebar-tab ${activeTab === 'ingest' ? 'active' : ''}`}
              >
                📤 Ingest PDF
              </button>
              <button
                onClick={() => setActiveTab('register-admin')}
                className={`sidebar-tab ${activeTab === 'register-admin' ? 'active' : ''}`}
              >
                🔑 Register Admin
              </button>
            </>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="user-avatar">{username?.substring(0, 2).toUpperCase()}</div>
            <div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{username}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{role?.toLowerCase()} profile</div>
            </div>
          </div>
          <button onClick={handleLogout} className="btn-logout" title="Log out">
            Logout
          </button>
        </div>
      </div>

      {/* Main Container */}
      <div className="main-workspace">
        {/* Toggle Sidebar Icon Button */}
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
          {sidebarOpen ? '◀' : '▶'} Menu
        </button>

        {/* Tab content conditional rendering */}
        {activeTab === 'chat' && (
          <div className="chat-workspace">
            {/* Timeline */}
            <div className="chat-history-wrapper">
              {messages.length === 0 ? (
                /* Welcome Centered Screen matching ChatGPT */
                <div className="welcome-container">
                  <img src="/image.png" alt="University Logo" className="welcome-logo" style={{ height: '60px', width: 'auto', borderRadius: '50%', marginBottom: '24px' }} onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                  <h1 className="welcome-title">Welcome to Prospectus AI</h1>
                  <p className="welcome-subtitle" style={{ maxWidth: '600px', margin: '0 auto', fontSize: '1.15rem', lineHeight: '1.7' }}>
                    Your official academic guide. 
                    Ask any questions about undergraduate or postgraduate admissions, eligibility criteria, department seats, 
                    course selections, or application procedures to begin your journey.
                  </p>
                </div>
              ) : (
                /* Conversation Timeline list */
                <div className="chat-history-content">
                  {messages.map((m, idx) => (
                    <div key={idx} className={`message-row ${m.role}`}>
                      <div className={`message-avatar ${m.role === 'user' ? 'user-avatar' : 'bot-avatar'}`}>
                        {m.role === 'user' ? username?.substring(0, 1).toUpperCase() : ''}
                      </div>
                      <div className="message-body-wrapper">
                        <div className="message-sender">
                          {m.role === 'user' ? 'You' : 'Prospectus AI'}
                        </div>
                        <div className="message-content markdown-body">
                          {m.role === 'assistant' && m.content === '' && isGenerating ? (
                            <div className="typing-indicator">
                              <div className="typing-dot"></div>
                              <div className="typing-dot"></div>
                              <div className="typing-dot"></div>
                            </div>
                          ) : (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                          )}
                        </div>
                        {m.role === 'assistant' && m.content !== '' && (idx !== messages.length - 1 || !isGenerating) && (
                          <div className="message-actions">
                            <button className="btn-action" onClick={() => copyToClipboard(m.content, idx)}>
                              {copiedIndex === idx ? '✓ Copied' : '📋 Copy'}
                            </button>
                            {m.content.includes('/seat_distribution.pdf') && (
                              <button className="btn-action" onClick={downloadSeatMatrix} style={{ color: 'var(--primary-accent)' }}>
                                📥 Download Seat Matrix PDF
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              )}
            </div>

            {/* Input pill form */}
            <div className="input-panel">
              <div className="input-container-inner">
                <form onSubmit={handleSendMessage} className="input-pill-wrapper">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Message Prospectus AI..."
                    className="input-pill-field"
                    disabled={isGenerating}
                  />
                  <button type="submit" className="btn-send-pill" disabled={isGenerating || !chatInput.trim()}>
                    ▲
                  </button>
                </form>
                <div className="input-disclaimer">
                  Prospectus AI can make mistakes. Please verify official schedules and criteria in the prospectus PDF.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Ingest PDF */}
        {activeTab === 'ingest' && (
          <div className="content-panel glass-panel">
            <h2>Ingest Prospectus PDF</h2>
            <p style={{ color: 'var(--text-muted)' }}>Process and index new undergraduate or postgraduate prospectus files into Pinecone database.</p>
            
            <form onSubmit={handleIngestSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '12px' }}>
              <div className="form-group" style={{ padding: '16px', background: 'rgba(59, 130, 246, 0.05)', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    id="usePgKnowledgeAdmin"
                    checked={usePgKnowledge}
                    onChange={(e) => setUsePgKnowledge(e.target.checked)}
                    style={{ margin: 0, width: '18px', height: '18px', cursor: 'pointer' }}
                  />
                  <label htmlFor="usePgKnowledgeAdmin" style={{ margin: 0, cursor: 'pointer', color: 'var(--text-main)', fontSize: '1rem', fontWeight: 600 }}>
                    Global: Enable Postgraduate Knowledge in Chat
                  </label>
                </div>
                <p style={{ margin: '8px 0 0 26px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  If disabled, the RAG pipeline will block postgraduate queries and fall back to undergraduate data.
                </p>
              </div>

              <div className="form-group">
                <label>Target Academic Level</label>
                <div style={{ display: 'flex', gap: '20px', marginTop: '8px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-main)' }}>
                    <input 
                      type="radio" 
                      name="acad_level" 
                      value="undergraduate" 
                      checked={academicLevel === 'undergraduate'} 
                      onChange={() => setAcademicLevel('undergraduate')} 
                      style={{ width: '16px', height: '16px' }}
                    />
                    Undergraduate Mode
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-main)' }}>
                    <input 
                      type="radio" 
                      name="acad_level" 
                      value="postgraduate" 
                      checked={academicLevel === 'postgraduate'} 
                      onChange={() => setAcademicLevel('postgraduate')} 
                      style={{ width: '16px', height: '16px' }}
                    />
                    Postgraduate Mode
                  </label>
                </div>
              </div>

              {academicLevel === 'undergraduate' && (
                <div className="form-group">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <input
                      type="checkbox"
                      id="extractSeats"
                      checked={extractSeats}
                      onChange={(e) => setExtractSeats(e.target.checked)}
                      style={{ margin: 0, width: '16px', height: '16px' }}
                    />
                    <label htmlFor="extractSeats" style={{ margin: 0, cursor: 'pointer' }}>
                      Extract Seat Distribution into Separate File
                    </label>
                  </div>
                  <label style={{ opacity: extractSeats ? 1 : 0.5 }}>Excluded Seat Distribution Pages (comma-separated, e.g. 79,80,81)</label>
                  <input
                    type="text"
                    value={excludedPages}
                    onChange={(e) => setExcludedPages(e.target.value)}
                    placeholder="e.g. 79,80,81"
                    disabled={!extractSeats}
                    required={extractSeats}
                    style={{ opacity: extractSeats ? 1 : 0.5 }}
                  />
                </div>
              )}

              <div className="form-group">
                <label>Select Prospectus PDF File</label>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="file-upload-input"
                  required
                />
              </div>

              {ingestStatus === 'uploading' && <div style={{ color: 'var(--primary-accent)', fontWeight: 600 }}>📤 Uploading file to backend...</div>}
              {ingestStatus === 'processing' && <div style={{ color: '#fbbf24', fontWeight: 600 }}>⚙️ {ingestMsg}</div>}
              {ingestStatus === 'success' && <div style={{ color: '#10b981', fontWeight: 600 }}>✅ {ingestMsg}</div>}
              {ingestStatus === 'error' && <div style={{ color: '#ef4444', fontWeight: 600 }}>⚠️ {ingestMsg}</div>}

              <button type="submit" className="btn-primary" style={{ alignSelf: 'flex-start' }} disabled={ingestStatus === 'uploading' || ingestStatus === 'processing'}>
                {ingestStatus === 'processing' ? 'Ingestion In Progress...' : 'Start Ingestion'}
              </button>
            </form>
          </div>
        )}

        {/* Tab 3: Register Admin */}
        {activeTab === 'register-admin' && (
          <div className="content-panel glass-panel">
            <h2>Register Admin Profile</h2>
            <p style={{ color: 'var(--text-muted)' }}>Generate a secondary credentials profile with full system modification rights.</p>
            
            <form onSubmit={handleRegisterAdmin} style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '12px' }}>
              <div className="form-group">
                <label>New Username</label>
                <input
                  type="text"
                  value={newAdminUser}
                  onChange={(e) => setNewAdminUser(e.target.value)}
                  placeholder="Enter username"
                  required
                />
              </div>
              <div className="form-group">
                <label>New Password</label>
                <input
                  type="password"
                  value={newAdminPass}
                  onChange={(e) => setNewAdminPass(e.target.value)}
                  placeholder="Enter password"
                  required
                />
              </div>

              {adminRegError && <div style={{ color: '#ef4444', fontSize: '0.9rem', fontWeight: 600 }}>⚠️ {adminRegError}</div>}
              {adminRegSuccess && <div style={{ color: '#10b981', fontSize: '0.9rem', fontWeight: 600 }}>✅ {adminRegSuccess}</div>}

              <button type="submit" className="btn-primary" style={{ alignSelf: 'flex-start' }}>
                Register Profile
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
