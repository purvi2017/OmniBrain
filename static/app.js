/**
 * OmniBrain RAG AI — Enterprise Integration Dashboard Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const API_BASE = '';

  // App State
  let currentSessionId = 'default_session';
  let selectedFiles = [];

  // Tab Navigation
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = document.getElementById(btn.dataset.tab);
      if (targetTab) targetTab.classList.add('active');

      // Refresh specific tab state on tab switch
      if (btn.dataset.tab === 'tab-sessions') loadSessions();
      if (btn.dataset.tab === 'tab-health') checkHealth();
    });
  });

  // Slider Values Synchronization
  const sliders = [
    { input: 'top-k', val: 'top-k-val' },
    { input: 'min-score', val: 'min-score-val' },
    { input: 'max-drop', val: 'max-drop-val' }
  ];

  sliders.forEach(s => {
    const el = document.getElementById(s.input);
    const label = document.getElementById(s.val);
    if (el && label) {
      el.addEventListener('input', () => { label.textContent = el.value; });
    }
  });

  // Toast Notifications
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // System Health Monitor
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      const data = await res.json();

      document.getElementById('metric-vectors').textContent = data.indexed_vectors;
      document.getElementById('metric-sessions').textContent = data.active_sessions;
      document.getElementById('metric-model').textContent = data.model;

      const statusPill = document.getElementById('status-pill');
      const statusText = document.getElementById('system-status-text');
      const statusDot = statusPill.querySelector('.status-dot');

      if (data.status === 'ok') {
        statusDot.classList.add('online');
        statusText.textContent = data.indexed_vectors > 0 ? 'System Online' : 'No Documents Indexed';
      }

      document.getElementById('health-json').textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      document.getElementById('system-status-text').textContent = 'API Unavailable';
      document.getElementById('health-json').textContent = `Error connecting to API: ${err.message}`;
    }
  }

  // File Upload / Ingestion Logic
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileList = document.getElementById('file-list');
  const btnIngest = document.getElementById('btn-ingest');

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
  });

  dropZone.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (files.length === 0) {
      showToast('Only .pdf files are accepted', 'error');
      return;
    }
    handleFilesSelected(files);
  });

  fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    handleFilesSelected(files);
  });

  function handleFilesSelected(files) {
    selectedFiles = files;
    fileList.innerHTML = '';
    
    files.forEach(f => {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `
        <span>📄 ${f.name}</span>
        <span class="file-size">${(f.size / 1024).toFixed(1)} KB</span>
      `;
      fileList.appendChild(item);
    });

    btnIngest.disabled = selectedFiles.length === 0;
  }

  btnIngest.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));

    btnIngest.disabled = true;
    btnIngest.innerHTML = '<span>⏳ Ingesting & Indexing...</span>';

    try {
      const res = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

      showToast(`Indexed ${data.documents.length} document(s) (${data.total_vectors} vectors)`, 'success');
      renderIngestResults(data);
      checkHealth();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnIngest.disabled = false;
      btnIngest.innerHTML = '<span>🚀 Process & Index Documents</span>';
    }
  });

  function renderIngestResults(data) {
    const resultsBox = document.getElementById('ingest-results');
    const summaryGrid = document.getElementById('ingest-summary');
    const tableBody = document.getElementById('ingest-table-body');

    resultsBox.classList.remove('hidden');
    summaryGrid.innerHTML = `
      <div class="metric-card">
        <span class="metric-label">Documents Ingested</span>
        <span class="metric-val">${data.documents.length}</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Total Chunks</span>
        <span class="metric-val">${data.total_chunks}</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Total FAISS Vectors</span>
        <span class="metric-val">${data.total_vectors}</span>
      </div>
    `;

    tableBody.innerHTML = data.documents.map(doc => `
      <tr>
        <td>📄 ${doc.filename}</td>
        <td><code>${doc.document_id}</code></td>
        <td>${doc.page_count}</td>
        <td>${doc.chunk_count}</td>
      </tr>
    `).join('');
  }

  // Single-Turn RAG Query Form
  const queryForm = document.getElementById('query-form');
  const queryResponseCard = document.getElementById('query-response-card');

  queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const queryText = document.getElementById('query-input').value.trim();
    const directContext = document.getElementById('direct-context').value.trim();
    const docIdFilter = document.getElementById('doc-id-filter').value.trim();
    const topK = parseInt(document.getElementById('top-k').value, 10);
    const minScore = parseFloat(document.getElementById('min-score').value);
    const maxDrop = parseFloat(document.getElementById('max-drop').value);

    if (!queryText) return;

    queryResponseCard.className = 'response-card placeholder-state';
    queryResponseCard.innerHTML = '<div class="placeholder-text"><span>⏳ Generating Grounded Answer...</span></div>';

    const payload = {
      query: queryText,
      context: directContext || null,
      document_id: docIdFilter || null,
      top_k: topK,
      min_score: minScore,
      max_score_drop: maxDrop
    };

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Query execution failed');

      renderQueryResponse(data);
    } catch (err) {
      queryResponseCard.className = 'response-card';
      queryResponseCard.innerHTML = `
        <div class="answer-header">
          <span class="badge badge-refused">Query Error</span>
        </div>
        <div class="answer-text" style="color: var(--accent-danger);">${err.message}</div>
      `;
      showToast(err.message, 'error');
    }
  });

  function renderQueryResponse(data) {
    queryResponseCard.className = 'response-card';
    const foundBadge = data.found 
      ? `<span class="badge badge-found">✔ Grounded Answer (${data.retrieved_chunks} Chunks)</span>`
      : `<span class="badge badge-refused">⚠ Refusal / No Context Found</span>`;

    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
      sourcesHtml = `
        <div class="sources-section">
          <h4>Attributed Source Documents (${data.sources.length})</h4>
          ${data.sources.map(s => `
            <div class="source-card">
              <div class="source-header">
                <span class="source-doc">📄 ${s.filename} (Page ${s.page})</span>
                <span class="source-score">Score: ${s.score} [${s.confidence}]</span>
              </div>
              <div class="source-preview">"${s.text_preview}"</div>
            </div>
          `).join('')}
        </div>
      `;
    }

    queryResponseCard.innerHTML = `
      <div class="answer-header">
        ${foundBadge}
        <span style="font-size: 12px; color: var(--text-muted); font-family: var(--font-mono);">${data.model}</span>
      </div>
      <div class="answer-text">${data.answer}</div>
      ${sourcesHtml}
    `;
  }

  // Conversational Chat Logic
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input-text');
  const chatMessages = document.getElementById('chat-messages');
  const currentSessionTitle = document.getElementById('current-session-title');
  const sessionTurnCounter = document.getElementById('session-turn-counter');
  const btnNewChat = document.getElementById('btn-new-chat');
  const btnClearHistory = document.getElementById('btn-clear-chat-history');

  btnNewChat.addEventListener('click', () => {
    currentSessionId = `session_${Date.now().toString(36)}`;
    currentSessionTitle.textContent = `Session: ${currentSessionId}`;
    sessionTurnCounter.textContent = '0 Turns';
    chatMessages.innerHTML = `
      <div class="chat-welcome">
        <div class="welcome-icon">💬</div>
        <h3>New Chat Session Started</h3>
        <p>Session ID: <code>${currentSessionId}</code></p>
      </div>
    `;
    loadSessions();
  });

  btnClearHistory.addEventListener('click', async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${currentSessionId}`, { method: 'DELETE' });
      if (res.ok) {
        showToast(`Cleared history for ${currentSessionId}`, 'success');
        chatMessages.innerHTML = '<div class="chat-welcome"><p>History cleared.</p></div>';
        sessionTurnCounter.textContent = '0 Turns';
        loadSessions();
      }
    } catch (err) {
      showToast(err.message, 'error');
    }
  });

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg) return;

    appendChatBubble('user', msg);
    chatInput.value = '';

    try {
      const res = await fetch(`${API_BASE}/chat/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Chat request failed');

      sessionTurnCounter.textContent = `${data.turn} Turn(s)`;
      appendChatBubble('assistant', data.answer, data.sources);
      loadSessions();
      checkHealth();
    } catch (err) {
      appendChatBubble('assistant', `⚠️ Error: ${err.message}`);
      showToast(err.message, 'error');
    }
  });

  function appendChatBubble(role, content, sources = []) {
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;

    let sourcesFooter = '';
    if (sources && sources.length > 0) {
      sourcesFooter = `
        <div style="margin-top: 8px; font-size: 11px; color: var(--accent-cyan); border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
          📍 Sources: ${sources.map(s => `${s.filename} (Page ${s.page})`).join(', ')}
        </div>
      `;
    }

    bubble.innerHTML = `<div>${content}</div>${sourcesFooter}`;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Session Explorer Table
  async function loadSessions() {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (!res.ok) return;
      const data = await res.json();

      const tbody = document.getElementById('sessions-table-body');
      const pillList = document.getElementById('chat-session-list');

      if (!data.sessions || data.sessions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center">No active chat sessions.</td></tr>';
        pillList.innerHTML = '<div style="font-size:12px; color:var(--text-muted);">No active sessions</div>';
        return;
      }

      pillList.innerHTML = data.sessions.map(s => `
        <div class="session-pill ${s.session_id === currentSessionId ? 'active' : ''}" data-sid="${s.session_id}">
          💬 ${s.session_id} (${s.turn_count} turns)
        </div>
      `).join('');

      tbody.innerHTML = data.sessions.map(s => `
        <tr>
          <td><code>${s.session_id}</code></td>
          <td>${s.turn_count}</td>
          <td>${s.message_count}</td>
          <td>
            <button class="btn btn-small btn-secondary btn-switch" data-sid="${s.session_id}">Switch</button>
            <button class="btn btn-small btn-danger btn-del" data-sid="${s.session_id}">Delete</button>
          </td>
        </tr>
      `).join('');

      // Add event listeners for session actions
      document.querySelectorAll('.session-pill, .btn-switch').forEach(el => {
        el.addEventListener('click', () => {
          currentSessionId = el.dataset.sid;
          currentSessionTitle.textContent = `Session: ${currentSessionId}`;
          showToast(`Switched to session ${currentSessionId}`, 'info');
          loadSessionHistory(currentSessionId);
        });
      });

      document.querySelectorAll('.btn-del').forEach(btn => {
        btn.addEventListener('click', async () => {
          const sid = btn.dataset.sid;
          await fetch(`${API_BASE}/sessions/${sid}`, { method: 'DELETE' });
          showToast(`Deleted session ${sid}`, 'info');
          loadSessions();
          checkHealth();
        });
      });
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }

  async function loadSessionHistory(sessionId) {
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();

      chatMessages.innerHTML = '';
      sessionTurnCounter.textContent = `${data.turn_count} Turn(s)`;

      data.messages.forEach(m => {
        appendChatBubble(m.role === 'user' ? 'user' : 'assistant', m.content, m.sources);
      });
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  document.getElementById('btn-refresh-sessions').addEventListener('click', loadSessions);

  // Initial Load
  checkHealth();
  loadSessions();
});
