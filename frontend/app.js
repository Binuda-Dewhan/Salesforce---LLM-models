/**
 * Salesforce AI Agent & LoRA CRM Intelligence Studio - Frontend Logic
 */

// Determine API Base URL dynamically (supports localhost or same-origin deployment)
const API_BASE = window.location.port === "8000" ? "" : "http://localhost:8000";

// DOM Elements
const chatStream = document.getElementById("chat-stream");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");
const rawNotesInput = document.getElementById("raw-notes-input");
const formattedNotesOutput = document.getElementById("formatted-notes-output");
const formatLoraBtn = document.getElementById("format-lora-btn");
const syncSfBtn = document.getElementById("sync-sf-btn");
const syncOppIdInput = document.getElementById("sync-opp-id");

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  if (tabName === "chat") {
    document.getElementById("tab-btn-chat").classList.add("active");
    document.getElementById("tab-chat").classList.add("active");
    chatInput.focus();
  } else {
    document.getElementById("tab-btn-lora").classList.add("active");
    document.getElementById("tab-lora").classList.add("active");
    rawNotesInput.focus();
  }
}

// Check Backend Health on Mount
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      console.log("[Health Check OK]", data);
    }
  } catch (err) {
    console.warn("[Backend Offline or CORS restricted]", err);
  }
}
checkHealth();

// Quick query button click
function useQuery(text) {
  switchTab("chat");
  chatInput.value = text;
  chatInput.focus();
}

function clearChat() {
  chatStream.innerHTML = `
    <div class="message agent">
      <div class="message-bubble">
        🧹 <strong>Chat Cleared!</strong> What would you like to explore in Salesforce today?
      </div>
      <div class="message-meta">Agent Brain • Ready</div>
    </div>
  `;
}

// Append a Message to Chat
function appendMessage(role, contentHtml, metaText = "Agent Brain") {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;

  const bubbleDiv = document.createElement("div");
  bubbleDiv.className = "message-bubble";
  bubbleDiv.innerHTML = contentHtml;

  const metaDiv = document.createElement("div");
  metaDiv.className = "message-meta";
  metaDiv.textContent = metaText;

  msgDiv.appendChild(bubbleDiv);
  msgDiv.appendChild(metaDiv);
  chatStream.appendChild(msgDiv);
  chatStream.scrollTop = chatStream.scrollHeight;
  return msgDiv;
}

// Format Records as Rich Table
function renderRecordsTable(records) {
  if (!records || records.length === 0) {
    return "<p style='color: var(--text-muted); font-size: 0.85rem;'>No matching records found in Salesforce.</p>";
  }

  let html = "<div style='overflow-x: auto;'><table class='sf-record-table'><thead><tr>";
  
  // Extract key columns
  const first = records[0];
  const keys = Object.keys(first).filter(k => k !== "attributes");

  keys.forEach(k => {
    html += `<th>${k}</th>`;
  });
  html += "</tr></thead><tbody>";

  records.forEach(row => {
    html += "<tr>";
    keys.forEach(k => {
      let val = row[k];
      if (k === "Amount" && val !== null && val !== undefined) {
        val = `$${Number(val).toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
      } else if (k === "StageName") {
        const isWon = String(val).toLowerCase().includes("won");
        val = `<span class="stage-badge ${isWon ? 'won' : ''}">${val}</span>`;
      }
      html += `<td>${val ?? '—'}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody></table></div>";
  return html;
}

// Lightweight Markdown to HTML Formatter for Grounded Responses
function formatMarkdown(text) {
  if (!text) return "";
  let html = text
    .replace(/^### (.*$)/gim, '<h4 style="margin: 0.8rem 0 0.35rem 0; color: #a5b4fc; font-size: 0.95rem;">$1</h4>')
    .replace(/^## (.*$)/gim, '<h3 style="margin: 0.9rem 0 0.45rem 0; color: #c7d2fe; font-size: 1.05rem;">$1</h3>')
    .replace(/^# (.*$)/gim, '<h2 style="margin: 1rem 0 0.55rem 0; color: #e0e7ff; font-size: 1.15rem;">$1</h2>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong style="color: #f8fafc;">$1</strong>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    .replace(/`([^`]+)`/gim, '<code style="background: rgba(0,0,0,0.3); padding: 0.15rem 0.35rem; border-radius: 4px; font-family: var(--font-mono); font-size: 0.82rem; color: #6ee7b7;">$1</code>')
    .replace(/^\s*[\-\*]\s+(.*$)/gim, '<li style="margin-left: 1.2rem; margin-bottom: 0.3rem;">$1</li>')
    .replace(/\n\n/gim, '<div style="margin-bottom: 0.6rem;"></div>')
    .replace(/\n/gim, '<br>');
  return html;
}

// Handle Conversational Chat Submission
async function handleChatSubmit(event) {
  event.preventDefault();
  const prompt = chatInput.value.trim();
  if (!prompt) return;

  // 1. Render User Message
  appendMessage("user", prompt, "You");
  chatInput.value = "";
  chatSendBtn.disabled = true;
  chatSendBtn.innerHTML = `<span class="spinner"></span>`;

  // Temporary loading bubble
  const loadingMsg = appendMessage("agent", `<em>Thinking, querying Salesforce & synthesizing...</em> <span class="spinner"></span>`, "CRM Executive Assistant");

  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE}/salesforce-chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt })
    });

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    loadingMsg.remove();

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    let replyHtml = "";

    // 1. New Grounded Agent Response Format
    if (data.type === "agent_response") {
      const isRefusal = data.is_refusal;
      const toolExecs = data.tool_executions || [];

      if (isRefusal) {
        replyHtml += `
          <div class="guardrail-refusal">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem; color: #fb7185; font-weight: 600; font-size: 0.85rem;">
              <span>🛡️ CRM Scope Guardrail</span>
            </div>
            <div style="font-size: 0.9rem; line-height: 1.5; color: #cbd5e1;">${formatMarkdown(data.answer)}</div>
          </div>
        `;
      } else {
        // Primary Grounded Executive Answer
        replyHtml += `<div class="agent-grounded-answer" style="font-size: 0.92rem; line-height: 1.6; color: #f1f5f9;">${formatMarkdown(data.answer)}</div>`;

        // Collapsible Grounded Audit Trail (if tools were executed)
        if (toolExecs.length > 0) {
          replyHtml += `
            <details class="tool-trace-drawer" style="margin-top: 0.85rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.5rem;">
              <summary style="cursor: pointer; font-size: 0.78rem; color: var(--text-muted); user-select: none; display: flex; align-items: center; gap: 0.4rem;">
                <span>⚡ Grounded in ${toolExecs.length} Salesforce Tool Execution(s)</span>
              </summary>
              <div style="margin-top: 0.6rem; display: flex; flex-direction: column; gap: 0.75rem;">
          `;

          toolExecs.forEach((te) => {
            const tool = te.tool;
            const args = te.args || {};
            const res = te.result || {};

            let badgeClass = "tool-badge";
            if (tool === "executeSOQL") badgeClass += " soql";
            if (tool === "formatNotes") badgeClass += " lora";

            replyHtml += `
              <div style="background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 0.6rem;">
                <div style="margin-bottom: 0.4rem;"><span class="${badgeClass}">Tool: ${tool}</span></div>
            `;

            if (tool === "executeSOQL") {
              replyHtml += `<div style="font-family: var(--font-mono); font-size: 0.75rem; color: #a5b4fc; background: rgba(0,0,0,0.3); padding: 0.35rem 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">${args.query || ''}</div>`;
              if (res.totalSize !== undefined && (!res.records || res.records.length === 0)) {
                replyHtml += `<p style="font-size: 0.82rem;"><strong>Total Records:</strong> <span style="color: #6ee7b7; font-weight: 700;">${res.totalSize}</span></p>`;
              } else if (res.records) {
                replyHtml += renderRecordsTable(res.records);
              }
            } else if (res.records) {
              replyHtml += renderRecordsTable(res.records);
            } else if (typeof res === "string") {
              replyHtml += `<div style="white-space: pre-wrap; font-size: 0.82rem;">${res}</div>`;
            } else {
              replyHtml += `<pre style="font-size: 0.75rem; color: var(--text-muted);">${JSON.stringify(res, null, 2)}</pre>`;
            }

            replyHtml += `</div>`;
          });

          replyHtml += `</div></details>`;
        }
      }

    // 2. Legacy / Direct Tool Result Handling (Backwards Compatible)
    } else if (data.type === "tool_result") {
      const tool = data.tool;
      const args = data.args;
      const res = data.result;

      let badgeClass = "tool-badge";
      if (tool === "executeSOQL") badgeClass += " soql";
      if (tool === "formatNotes") badgeClass += " lora";

      replyHtml += `<div><span class="${badgeClass}">⚡ Tool Executed: ${tool}</span></div>`;

      if (tool === "executeSOQL") {
        replyHtml += `<div style="font-family: var(--font-mono); font-size: 0.78rem; color: #a5b4fc; background: rgba(0,0,0,0.3); padding: 0.4rem 0.6rem; border-radius: 4px; margin-bottom: 0.6rem;">${args.query}</div>`;
        if (res.totalSize !== undefined && (!res.records || res.records.length === 0)) {
          replyHtml += `<p><strong>Query Count Result:</strong> <span style="font-size: 1.1rem; color: #6ee7b7; font-weight: 700;">${res.totalSize} records</span></p>`;
        } else if (res.records) {
          replyHtml += renderRecordsTable(res.records);
        }
      } else if (tool === "searchOpportunities" || tool === "searchAccounts") {
        const records = res.records || [];
        replyHtml += `<p style="margin-bottom: 0.4rem;">Found <strong>${records.length}</strong> matching record(s):</p>`;
        replyHtml += renderRecordsTable(records);
      } else if (tool === "getOpportunity") {
        const records = res.records || [];
        replyHtml += renderRecordsTable(records);
      } else if (tool === "createNotes") {
        replyHtml += `<p style="color: #6ee7b7; font-weight: 600;">✅ Note successfully attached to Opportunity ${args.opportunityId} in Salesforce!</p>`;
        replyHtml += `<pre style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.4rem;">${args.newNoteToAdd}</pre>`;
      } else if (tool === "formatNotes") {
        replyHtml += `<div style="white-space: pre-wrap; font-size: 0.88rem; line-height: 1.6;">${res}</div>`;
      } else {
        replyHtml += `<pre style="font-size: 0.8rem;">${JSON.stringify(res, null, 2)}</pre>`;
      }

    } else if (data.type === "answer") {
      replyHtml = `<div style="white-space: pre-wrap;">${formatMarkdown(data.message)}</div>`;
    } else if (data.type === "error") {
      replyHtml = `<div style="color: var(--accent-rose);">❌ <strong>Error:</strong> ${data.error}</div>`;
    } else {
      replyHtml = `<pre style="font-size: 0.8rem;">${JSON.stringify(data, null, 2)}</pre>`;
    }

    appendMessage("agent", replyHtml, `Agent Brain • ${elapsed}s`);

  } catch (error) {
    loadingMsg.remove();
    appendMessage("agent", `<div style="color: var(--accent-rose);">⚠️ <strong>Network/Server Error:</strong> ${error.message}</div>`, "System Error");
  } finally {
    chatSendBtn.disabled = false;
    chatSendBtn.innerHTML = `<span>Send</span>`;
  }
}

// Sample Raw Meeting Notes
const SAMPLES = [
  "RetailAxis - Merchandising Lead Sarah Gold: Struggling with inventory sync between Shopify and warehouse. Legacy system takes 4 minutes per SKU. Wants a comprehensive cloud migration plan by Q1. Needs support for 500 store managers. Action: Will send existing inventory reports by Friday. Follow up next Tuesday.",
  "Acme Corp - CTO John Miller: Legacy CRM takes 5 minutes per update. Experiencing data sync latency. Interested in reducing manual data entry by 40%. Budget approved for Q4. Action: Send architecture blueprint and schedule technical deep dive next week.",
  "United Oil - VP Operations David Vance: Current vendor comparison underway. Requires scaling to 5,000 users. Compliance security audit required. Action: Send security compliance whitepaper and schedule team demo."
];

let sampleIdx = 0;
function loadSampleNotes() {
  rawNotesInput.value = SAMPLES[sampleIdx % SAMPLES.length];
  sampleIdx++;
}

// Format Notes directly via LoRA endpoint
async function formatNotesDirect() {
  const raw = rawNotesInput.value.trim();
  if (!raw) {
    alert("Please enter or paste raw meeting notes first.");
    rawNotesInput.focus();
    return;
  }

  formatLoraBtn.disabled = true;
  formatLoraBtn.innerHTML = `<span class="spinner"></span> <span>Formatting via LoRA...</span>`;
  formattedNotesOutput.innerHTML = `<div style="display: flex; align-items: center; gap: 0.5rem; color: #a5b4fc;"><span class="spinner"></span> Running on-device 4-bit Gemma-2B LoRA inference in GPU VRAM...</div>`;

  try {
    const res = await fetch(`${API_BASE}/format-notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_notes: raw })
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    const formatted = data.formatted_notes;
    formattedNotesOutput.textContent = formatted;

  } catch (err) {
    formattedNotesOutput.innerHTML = `<span style="color: var(--accent-rose);">❌ Formatting failed: ${err.message}</span>`;
  } finally {
    formatLoraBtn.disabled = false;
    formatLoraBtn.innerHTML = `<span>✨ Format with Gemma-2B LoRA</span>`;
  }
}

function copyFormatted() {
  const text = formattedNotesOutput.textContent;
  if (!text || text.includes("Click \"Format with Gemma-2B LoRA\"")) return;
  navigator.clipboard.writeText(text);
  alert("Formatted CRM notes copied to clipboard!");
}

// Sync Formatted Note to Salesforce
async function syncNoteToSalesforce() {
  const noteBody = formattedNotesOutput.textContent.trim();
  const oppId = syncOppIdInput.value.trim();

  if (!noteBody || noteBody.includes("Click \"Format with Gemma-2B LoRA\"")) {
    alert("Please format a note first before syncing to Salesforce.");
    return;
  }
  if (!oppId) {
    alert("Please provide a valid Opportunity ID.");
    syncOppIdInput.focus();
    return;
  }

  syncSfBtn.disabled = true;
  syncSfBtn.innerHTML = `<span class="spinner"></span> <span>Syncing to Cloud...</span>`;

  try {
    const prompt = `Create a note on ${oppId} saying: ${noteBody}`;
    const res = await fetch(`${API_BASE}/salesforce-chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt })
    });

    const data = await res.json();
    if (res.ok && data.type === "tool_result" && data.tool === "createNotes") {
      alert(`🎉 Successfully synced note to Opportunity ${oppId} in Salesforce Cloud!`);
      switchTab("chat");
      appendMessage("agent", `✅ <strong>Live Sync Confirmed:</strong> Note attached to Opportunity <code>${oppId}</code> in Salesforce Cloud.<br><pre style="font-size: 0.8rem; margin-top: 0.4rem;">${noteBody}</pre>`, "Salesforce REST Sync");
    } else {
      throw new Error(data.error || "Failed to create note in Salesforce.");
    }
  } catch (err) {
    alert(`Error syncing to Salesforce: ${err.message}`);
  } finally {
    syncSfBtn.disabled = false;
    syncSfBtn.innerHTML = `<span>☁️ Sync Note to Salesforce</span>`;
  }
}
