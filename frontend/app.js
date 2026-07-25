/* ═══════════════════════════════════════════════════
   AML SENTINEL 2.0 — Frontend Application Logic
   Handles chat, API communication, chart rendering,
   execution trace display, and Lucide SVG injection
   ═══════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── State ──
let isProcessing = false;

// ── DOM Elements ──
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.getElementById('chat-history');
const resultsContent = document.getElementById('results-content');
const executionSummary = document.getElementById('execution-summary');
const traceList = document.getElementById('trace-list');
const entityList = document.getElementById('entity-list');
const intentDisplay = document.getElementById('intent-display');

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    checkHealth();
    loadDatasetInfo();
    setupChips();
    setupForm();
});

// ── Health Check ──
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const dot = document.getElementById('status-dot');
        const label = document.getElementById('health-status');
        
        if (data.status === 'healthy' && data.data_loaded) {
            dot.classList.remove('error');
            label.textContent = `ONLINE [${data.transactions_count.toLocaleString()} TXNS]`;
        } else {
            dot.classList.add('error');
            label.textContent = 'DATA UNAVAILABLE';
        }
    } catch (e) {
        const dot = document.getElementById('status-dot');
        const label = document.getElementById('health-status');
        dot.classList.add('error');
        label.textContent = 'OFFLINE';
    }
}

// ── Dataset Info ──
async function loadDatasetInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/dataset/info`);
        const data = await res.json();
        const meta = document.getElementById('dataset-meta');
        meta.textContent = `${data.total_transactions?.toLocaleString() || '?'} TXNS`;
    } catch (e) {
        document.getElementById('dataset-meta').textContent = 'N/A';
    }
}

// ── Example Chips ──
function setupChips() {
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            queryInput.value = query;
            submitQuery(query);
        });
    });
}

// ── Form Submission ──
function setupForm() {
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (query && !isProcessing) {
            submitQuery(query);
        }
    });
}

// ── Main Query Submission ──
async function submitQuery(query) {
    if (isProcessing) return;
    isProcessing = true;
    sendBtn.disabled = true;
    queryInput.value = '';

    // Add user message
    addMessage('user', query);
    
    // Show typing indicator
    const typingEl = addTypingIndicator();
    
    // Clear previous results
    clearTrace();
    
    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await res.json();
        
        // Remove typing indicator
        typingEl.remove();
        
        // Add agent response message
        addMessage('agent', `Analysis complete. Analyzed ${data.total_analyzed?.toLocaleString() || 0} transactions, flagged ${data.total_flagged || 0} entities in ${(data.processing_time_ms || 0).toFixed(0)}ms.`);
        
        // Render results
        renderResults(data);
        
    } catch (err) {
        typingEl.remove();
        addMessage('agent', `Error: ${err.message}`);
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        queryInput.focus();
        lucide.createIcons(); // re-initialize icons for dynamically added elements
    }
}

// ── Message Rendering ──
function addMessage(type, content) {
    const div = document.createElement('div');
    const icon = type === 'user' ? 'user' : 'cpu';
    const className = type === 'user' ? 'user-msg' : 'agent-msg';
    
    div.className = `message ${className}`;
    div.innerHTML = `
        <div class="msg-icon"><i data-lucide="${icon}"></i></div>
        <div class="msg-content">${content}</div>
    `;
    
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    lucide.createIcons({root: div});
    return div;
}

function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message agent-msg typing-indicator';
    div.innerHTML = `
        <div class="msg-icon"><i data-lucide="cpu"></i></div>
        <div class="msg-content" style="display:flex;gap:8px;align-items:center;">
            <div class="typing-pulse">
                <div></div><div></div><div></div>
            </div>
            <span style="color:var(--text-muted);font-size:11px;letter-spacing:1px;text-transform:uppercase;">Processing</span>
        </div>
    `;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    lucide.createIcons({root: div});
    return div;
}

// ── Results Rendering ──
function renderResults(data) {
    resultsContent.innerHTML = '';
    
    // Show execution summary bar
    executionSummary.style.display = 'flex';
    animateNumber('total-analyzed', data.total_analyzed || 0);
    document.getElementById('total-flagged-center').textContent = data.total_flagged || 0;
    document.getElementById('processing-time').textContent = `${(data.processing_time_ms || 0).toFixed(0)}ms`;
    
    // Update intent
    intentDisplay.textContent = (data.intent_detected || '—').replace(/_/g, ' ');
    
    // Render execution trace
    renderTrace(data.execution_trace || []);
    
    // Render summary
    if (data.summary) {
        const section = createResultSection('Analysis Summary', 'result-text');
        // Clean any residual emojis in backend response just in case
        section.querySelector('.result-text').textContent = data.summary.replace(/[\u{1F300}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '');
        resultsContent.appendChild(section);
    }
    
    // Render charts
    if (data.charts && data.charts.length > 0) {
        const section = document.createElement('div');
        section.className = 'result-section';
        section.innerHTML = `<div class="result-section-title"><i data-lucide="pie-chart"></i> Visualizations (${data.charts.length})</div>`;
        
        const grid = document.createElement('div');
        grid.className = 'chart-grid';
        
        data.charts.forEach((chartB64, i) => {
            const card = document.createElement('div');
            card.className = 'chart-card';
            card.style.animationDelay = `${i * 0.1}s`;
            card.innerHTML = `<img src="data:image/png;base64,${chartB64}" alt="Chart ${i+1}" loading="lazy">`;
            grid.appendChild(card);
        });
        
        section.appendChild(grid);
        resultsContent.appendChild(section);
    }
    
    // Render flagged entities table
    if (data.flagged_entities && data.flagged_entities.length > 0) {
        renderFlaggedTable(data.flagged_entities);
        renderEntityList(data.flagged_entities.slice(0, 8));
        updateRiskCards(data.flagged_entities);
    } else {
        updateRiskCards([]);
    }
}

function createResultSection(title, contentClass) {
    const section = document.createElement('div');
    section.className = 'result-section';
    section.innerHTML = `
        <div class="result-section-title"><i data-lucide="file-text"></i> ${title}</div>
        <div class="${contentClass || ''}"></div>
    `;
    return section;
}

// ── Flagged Entities Table ──
function renderFlaggedTable(entities) {
    const section = document.createElement('div');
    section.className = 'result-section';
    section.innerHTML = `
        <div class="result-section-title"><i data-lucide="shield-alert"></i> Flagged Entities (${entities.length})</div>
        <div class="results-table-container">
            <table class="results-table" id="flagged-table">
                <thead>
                    <tr>
                        <th data-sort="entity_id">Entity ID</th>
                        <th data-sort="risk_score">Risk Score ↕</th>
                        <th data-sort="risk_level">Risk Level</th>
                        <th>Action</th>
                        <th>Patterns</th>
                        <th>Explanation</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    `;
    
    const tbody = section.querySelector('tbody');
    
    entities.forEach((e, i) => {
        const tr = document.createElement('tr');
        tr.style.animationDelay = `${i * 0.05}s`;
        tr.className = 'fadeSlideIn';
        
        const riskClass = e.risk_level || 'medium';
        const actionLabel = e.escalation_action === 'file_sar' ? 'FILE SAR' :
                           e.escalation_action === 'flag_for_review' ? 'REVIEW' : 'MONITOR';
        
        const patterns = (e.detected_patterns || []).join(', ') || '—';
        const explanation = (e.explanation || '').split('\n').slice(0, 2).join(' ').substring(0, 120);
        
        tr.innerHTML = `
            <td class="mono" style="color:var(--text-primary)">${e.entity_id}</td>
            <td class="mono" style="font-weight:700">${(e.risk_score || 0).toFixed(3)}</td>
            <td><span class="risk-badge ${riskClass}">${riskClass}</span></td>
            <td><span class="risk-badge ${riskClass}" style="background:transparent;border:1px solid currentColor">${actionLabel}</span></td>
            <td style="font-size:11px;text-transform:uppercase">${patterns.replace(/_/g, ' ')}</td>
            <td style="font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${explanation}...</td>
        `;
        
        tbody.appendChild(tr);
    });
    
    resultsContent.appendChild(section);
}

// ── Execution Trace ──
function clearTrace() {
    traceList.innerHTML = '<div class="trace-empty">Executing pipeline...</div>';
    entityList.innerHTML = '<div class="entity-empty">Awaiting results...</div>';
}

function renderTrace(trace) {
    traceList.innerHTML = '';
    
    const toolIcons = {
        'eda_tool': 'bar-chart',
        'feature_engineering': 'git-branch',
        'anomaly_detection': 'radar',
        'risk_classification': 'shield',
        'explanation_engine': 'message-square',
    };
    
    const toolLabels = {
        'eda_tool': 'EDA',
        'feature_engineering': 'Feature Eng.',
        'anomaly_detection': 'Anomaly Det.',
        'risk_classification': 'Risk Class.',
        'explanation_engine': 'Explanation',
    };
    
    trace.forEach((step, i) => {
        setTimeout(() => {
            const div = document.createElement('div');
            div.className = `trace-step ${step.status}`;
            
            const statusIcon = step.status === 'completed' ? 'check' : 
                              step.status === 'skipped' ? 'fast-forward' : 
                              step.status === 'error' ? 'x' : 'loader';
            
            div.innerHTML = `
                <span class="trace-icon"><i data-lucide="${toolIcons[step.tool_name] || 'settings'}"></i></span>
                <span class="trace-name">${toolLabels[step.tool_name] || step.tool_name}</span>
                <span class="trace-time">${step.duration_ms ? step.duration_ms.toFixed(0) + 'ms' : '—'}</span>
                <span class="trace-status"><i data-lucide="${statusIcon}"></i></span>
            `;
            
            traceList.appendChild(div);
            lucide.createIcons({root: div});
        }, i * 200);
    });
}

// ── Entity List (Right Panel) ──
function renderEntityList(entities) {
    entityList.innerHTML = '';
    
    if (entities.length === 0) {
        entityList.innerHTML = '<div class="entity-empty">No entities flagged.</div>';
        return;
    }
    
    entities.forEach((e, i) => {
        const div = document.createElement('div');
        div.className = 'entity-item';
        div.style.animationDelay = `${i * 0.1}s`;
        
        const riskClass = e.risk_level || 'medium';
        const color = riskClass === 'critical' ? 'var(--red)' : 
                     riskClass === 'high' ? 'var(--orange)' : 
                     riskClass === 'medium' ? 'var(--amber)' : 'var(--green)';
        
        div.innerHTML = `
            <span style="display:flex;align-items:center;">
                <span class="entity-indicator" style="background:${color};box-shadow:0 0 6px ${color}"></span>
                <span class="entity-id">${e.entity_id}</span>
            </span>
            <span class="entity-score" style="color:${color}">${(e.risk_score || 0).toFixed(3)}</span>
        `;
        
        entityList.appendChild(div);
    });
}

// ── Risk Cards ──
function updateRiskCards(entities) {
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };
    
    entities.forEach(e => {
        const level = e.risk_level || 'low';
        if (counts[level] !== undefined) counts[level]++;
    });
    
    animateNumber('risk-critical', counts.critical);
    animateNumber('risk-high', counts.high);
    animateNumber('risk-medium', counts.medium);
    animateNumber('risk-low', counts.low);
}

// ── Number Animation ──
function animateNumber(elementId, target) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    const start = parseInt(el.textContent) || 0;
    const duration = 600;
    const startTime = Date.now();
    
    function update() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (target - start) * eased);
        el.textContent = current.toLocaleString();
        
        if (progress < 1) requestAnimationFrame(update);
    }
    
    requestAnimationFrame(update);
}
