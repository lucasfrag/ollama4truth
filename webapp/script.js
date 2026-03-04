const form = document.getElementById("claimForm");
const input = document.getElementById("claimInput");
const modeSelect = document.getElementById("modeSelect");
const retrievalMethodSelect = document.getElementById("retrievalMethodSelect");
const strategySelect = document.getElementById("strategySelect");
const modelSelect = document.getElementById("modelSelect");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const sidebar = document.getElementById("sidebar");
const questionList = document.getElementById("questionList");

let currentData = null;

// Fetch available Ollama models on page load
async function fetchModels() {
    try {
        const res = await fetch("/models");
        const data = await res.json();
        modelSelect.innerHTML = "";
        data.models.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.name;
            opt.textContent = m.size ? `${m.name} (${m.size})` : m.name;
            if (m.name === data.default) opt.selected = true;
            modelSelect.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to fetch models:", err);
        modelSelect.innerHTML = '<option value="">Erro ao carregar</option>';
    }
}
fetchModels();

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const claim = input.value.trim();
    if (!claim) return;

    const mode = modeSelect.value;
    const retrievalMethod = retrievalMethodSelect.value;
    const strategy = strategySelect.value;
    const ollamaModel = modelSelect.value;

    // Reset UI
    result.innerHTML = "";
    sidebar.classList.add("hidden");
    result.classList.add("hidden");
    loading.classList.remove("hidden");
    questionList.innerHTML = "";
    document.getElementById("summary").innerHTML = "";

    // Cria conexão SSE with all params including model
    const params = new URLSearchParams({ claim, mode, strategy, retrieval_method: retrievalMethod, ollama_model: ollamaModel });
    const eventSource = new EventSource(
        "/analyze-stream?" + params
    );

    let finalResult = null;

    eventSource.onmessage = (event) => {
        const msg = event.data;

        // Recebe JSON final
        if (msg.startsWith("{")) {
            finalResult = JSON.parse(msg);
            eventSource.close();

            // Atualiza estado global
            currentData = finalResult;

            loading.classList.add("hidden");
            sidebar.classList.remove("hidden");
            result.classList.remove("hidden");

            console.log("🔍 Resultado final:", finalResult);
            renderClaim(finalResult);
            renderQuestionsAndAnswers(finalResult);
            return;
        }

        // Logs intermediários
        loading.textContent = msg;
    };

    eventSource.onerror = () => {
        loading.textContent = "❌ Erro ao receber dados em tempo real.";
        eventSource.close();
    };
});

// =============================
// 🔹 Renderiza a claim principal
// =============================
function renderClaim(data) {
    result.innerHTML = "";

    // Claim
    const claimEl = document.createElement("div");
    claimEl.classList.add("claim-title");
    claimEl.textContent = "📝 " + data.claim;
    result.appendChild(claimEl);

    // Timestamp
    if (data.timestamp) {
        const tsEl = document.createElement("div");
        tsEl.classList.add("claim-timestamp");
        const date = new Date(data.timestamp);
        tsEl.textContent = `Analisado em: ${date.toLocaleString()}`;
        result.appendChild(tsEl);
    }

    // Mode & Strategy info
    const infoEl = document.createElement("div");
    infoEl.classList.add("analysis-info");
    const modeLabel = { rag: "RAG (Corpus Local)", web: "Web (Google)", hybrid: "Híbrido" }[data.mode] || data.mode;
    const retrievalLabel = { bm25: "BM25", semantic: "Semântico", hybrid: "Híbrido" }[data.retrieval_method] || data.retrieval_method;
    const stratLabel = { ollama_verdict: "Ollama LLM", label_vote: "Votação por Labels" }[data.strategy] || data.strategy;
    infoEl.textContent = `Modo: ${modeLabel} | Recuperação: ${retrievalLabel} | Estratégia: ${stratLabel}`;
    result.appendChild(infoEl);

    // Label
    const labelEl = document.createElement("span");
    labelEl.classList.add("label-badge");
    if (data.label) {
        labelEl.classList.add(
            data.label.toUpperCase() === "APOIADA" ? "label-supported" :
                data.label.toUpperCase() === "REFUTADA" ? "label-refuted" :
                    "label-uncertain"
        );
        labelEl.textContent = data.label.toUpperCase();
    } else {
        labelEl.classList.add("label-uncertain");
        labelEl.textContent = "Sem evidências suficientes";
    }
    result.appendChild(labelEl);

    // Confiança (consistency-based)
    if (data.confidence !== null && data.confidence !== undefined) {
        const confEl = document.createElement("div");
        confEl.classList.add("confidence");
        confEl.style.marginLeft = "5px";
        if (data.consistency_detail) {
            const agree = data.consistency_detail.filter(v => v === data.label).length;
            confEl.textContent = `Consistência: ${data.confidence}% (${agree}/${data.consistency_detail.length})`;
        } else {
            confEl.textContent = `Confiança: ${data.confidence}%`;
        }
        result.appendChild(confEl);
    }

    // Rationale
    if (data.rationale) {
        const rationaleEl = document.createElement("p");
        rationaleEl.classList.add("rationale");
        rationaleEl.textContent = "💡 " + data.rationale;
        rationaleEl.style.fontWeight = "bold";
        result.appendChild(rationaleEl);
    }
}

// ================================================
// 🔹 Renderiza perguntas com respostas e evidências
// ================================================
function renderQuestionsAndAnswers(data) {
    const evidences = data.evidences || [];
    questionList.innerHTML = "";

    if (evidences.length === 0) {
        questionList.innerHTML = "<li>Nenhuma pergunta gerada.</li>";
        return;
    }

    // Sidebar: list of questions (clickable to scroll)
    evidences.forEach((ev, idx) => {
        const li = document.createElement("li");
        li.textContent = `${ev.question} (${(ev.results || []).length} evidências)`;
        if (idx === 0) li.classList.add("active");
        li.addEventListener("click", () => {
            document.querySelectorAll("#questionList li").forEach(el => el.classList.remove("active"));
            li.classList.add("active");
            // Scroll to the corresponding card
            const card = document.getElementById(`qa-card-${idx}`);
            if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
        });
        questionList.appendChild(li);
    });

    // Main panel: Q&A cards with evidence
    evidences.forEach((ev, idx) => {
        const card = document.createElement("div");
        card.classList.add("evidence-card");
        card.id = `qa-card-${idx}`;

        // Question header
        const qHeader = document.createElement("h3");
        qHeader.textContent = `❓ ${ev.question}`;
        card.appendChild(qHeader);

        // Answer
        if (ev.answer) {
            const answerEl = document.createElement("div");
            answerEl.classList.add("rationale");
            answerEl.innerHTML = `<strong>💬 Resposta:</strong> ${ev.answer}`;
            card.appendChild(answerEl);
        }

        // Evidence articles
        if (ev.results && ev.results.length > 0) {
            const evHeader = document.createElement("p");
            evHeader.style.cssText = "font-size:0.85rem;color:#94a3b8;margin-top:12px;margin-bottom:6px;";
            evHeader.textContent = `📰 ${ev.results.length} artigos consultados:`;
            card.appendChild(evHeader);

            const ul = document.createElement("ul");
            ev.results.forEach(res => {
                const li = document.createElement("li");
                let sourceInfo = "";
                if (res.source) {
                    sourceInfo = `<span class="source-badge">${res.source}</span> `;
                }
                if (res.label) {
                    sourceInfo += `<span class="label-badge-small label-${res.label.includes('fals') || res.label.includes('fake') || res.label.includes('enganoso') ? 'refuted' : res.label.includes('verdade') || res.label.includes('fato') ? 'supported' : 'uncertain'}">${res.label}</span> `;
                }
                li.innerHTML = `${sourceInfo}<a href="${res.link}" target="_blank">${res.title}</a>`;
                if (res.score != null) li.innerHTML += ` <span class="history-score">(${res.score.toFixed(4)})</span>`;
                ul.appendChild(li);
            });
            card.appendChild(ul);
        } else {
            const noEv = document.createElement("p");
            noEv.style.cssText = "font-size:0.8rem;color:#64748b;font-style:italic;";
            noEv.textContent = "Nenhuma evidência encontrada para esta pergunta.";
            card.appendChild(noEv);
        }

        result.appendChild(card);
    });

    // Update summary
    renderSummary();
}

// ===============================
// 🔹 Renderiza o resumo lateral
// ===============================
function renderSummary() {
    const summaryDiv = document.getElementById("summary");
    summaryDiv.innerHTML = "<h3>Resumo de Evidências</h3>";
    const evidences = currentData?.evidences || [];

    if (evidences.length === 0) return;

    const ul = document.createElement("ul");
    evidences.forEach(ev => {
        const li = document.createElement("li");
        li.textContent = `${ev.question} → ${(ev.results || []).length} evidências`;
        ul.appendChild(li);
    });
    summaryDiv.appendChild(ul);
}

// ========================================
// 🔹 History Panel
// ========================================
let historyVisible = false;

async function toggleHistory() {
    const panel = document.getElementById("historyPanel");
    const clearBtn = document.getElementById("clearHistory");

    if (historyVisible) {
        panel.classList.add("hidden");
        clearBtn.classList.add("hidden");
        historyVisible = false;
        return;
    }

    try {
        const res = await fetch("/history");
        const data = await res.json();

        document.getElementById("historyCount").textContent = `(${data.count} análise${data.count !== 1 ? "s" : ""})`;

        const historyList = document.getElementById("historyList");
        historyList.innerHTML = "";

        if (data.entries.length === 0) {
            historyList.innerHTML = "<p class='history-empty'>Nenhuma análise registrada ainda.</p>";
        } else {
            data.entries.forEach((entry, idx) => {
                historyList.appendChild(renderHistoryEntry(entry, idx + 1));
            });
        }

        panel.classList.remove("hidden");
        clearBtn.classList.remove("hidden");
        historyVisible = true;
    } catch (err) {
        console.error("Erro ao carregar histórico:", err);
    }
}

async function clearHistory() {
    if (!confirm("Limpar todo o histórico de análises?")) return;
    try {
        await fetch("/history/clear");
        document.getElementById("historyList").innerHTML = "<p class='history-empty'>Histórico limpo.</p>";
        document.getElementById("historyCount").textContent = "(0 análises)";
    } catch (err) {
        console.error("Erro ao limpar histórico:", err);
    }
}

function renderHistoryEntry(entry, num) {
    const card = document.createElement("div");
    card.classList.add("history-card");

    const labelClass = entry.label
        ? entry.label.toUpperCase() === "APOIADA" ? "label-supported"
            : entry.label.toUpperCase() === "REFUTADA" ? "label-refuted"
                : "label-uncertain"
        : "label-uncertain";

    const modeLabel = { rag: "RAG", web: "Web", hybrid: "Híbrido" }[entry.mode] || entry.mode;
    const retrievalLabel = { bm25: "BM25", semantic: "Semântico", hybrid: "Híbrido" }[entry.retrieval_method] || entry.retrieval_method || "bm25";
    const stratLabel = { ollama_verdict: "Ollama LLM", label_vote: "Votação por Rótulos" }[entry.strategy] || entry.strategy;

    // Header
    const header = document.createElement("div");
    header.classList.add("history-header");
    header.innerHTML = `
        <div class="history-header-top">
            <span class="history-index">#${num}</span>
            <span class="label-badge ${labelClass}">${(entry.label || "N/A").toUpperCase()}</span>
            ${entry.confidence != null ? `<span class="confidence">${entry.consistency_detail ? `${entry.confidence}% (${entry.consistency_detail.filter(v => v === entry.label).length}/${entry.consistency_detail.length})` : `${entry.confidence}%`}</span>` : ""}
            <span class="history-timestamp">${new Date(entry.timestamp).toLocaleString()}</span>
        </div>
        <div class="history-claim">${entry.claim}</div>
        <div class="history-settings">${modeLabel} · ${retrievalLabel} · ${stratLabel} · 🤖 ${entry.ollama_model || "unknown"}</div>
    `;
    header.style.cursor = "pointer";

    // Body (expandable)
    const body = document.createElement("div");
    body.classList.add("history-body", "hidden");

    // Rationale
    if (entry.rationale) {
        const rat = document.createElement("div");
        rat.classList.add("rationale");
        rat.textContent = "💡 " + entry.rationale;
        body.appendChild(rat);
    }

    // Q&A with Evidence (combined view)
    const evidences = entry.evidences || [];
    if (evidences.length > 0) {
        const sec = document.createElement("div");
        sec.classList.add("history-section");

        const totalArticles = evidences.reduce((sum, ev) => sum + (ev.results || []).length, 0);
        sec.innerHTML = `<h4>🧩 Perguntas, Respostas & Evidências (${evidences.length} perguntas, ${totalArticles} artigos)</h4>`;

        evidences.forEach((ev, idx) => {
            // Question
            const qDiv = document.createElement("div");
            qDiv.classList.add("history-evidence-question");
            qDiv.textContent = `${idx + 1}. ${ev.question}`;
            sec.appendChild(qDiv);

            // Answer
            if (ev.answer) {
                const ansDiv = document.createElement("div");
                ansDiv.style.cssText = "font-size:0.85rem;color:#93c5fd;margin:4px 0 6px 20px;padding:6px 10px;background:#162544;border-radius:6px;border:1px solid #1e3a5f;";
                ansDiv.innerHTML = `<strong>💬</strong> ${ev.answer}`;
                sec.appendChild(ansDiv);
            }

            // Evidence articles
            if (ev.results && ev.results.length > 0) {
                const ul = document.createElement("ul");
                ev.results.forEach(r => {
                    const li = document.createElement("li");
                    let info = "";
                    if (r.source) info += `<span class="source-badge">${r.source}</span> `;
                    if (r.label) {
                        const lc = r.label.includes("fals") || r.label.includes("fake") || r.label.includes("enganoso")
                            ? "refuted" : r.label.includes("verdade") || r.label.includes("fato") ? "supported" : "uncertain";
                        info += `<span class="label-badge-small label-${lc}">${r.label}</span> `;
                    }
                    li.innerHTML = `${info}<a href="${r.link}" target="_blank">${r.title}</a>`;
                    if (r.score != null) li.innerHTML += ` <span class="history-score">(${r.score.toFixed(4)})</span>`;
                    ul.appendChild(li);
                });
                sec.appendChild(ul);
            } else {
                const noEv = document.createElement("p");
                noEv.style.cssText = "font-size:0.8rem;color:#64748b;font-style:italic;margin:2px 0 8px 20px;";
                noEv.textContent = "Nenhuma evidência encontrada.";
                sec.appendChild(noEv);
            }
        });

        body.appendChild(sec);
    }

    header.addEventListener("click", () => {
        body.classList.toggle("hidden");
        card.classList.toggle("expanded");
    });

    card.appendChild(header);
    card.appendChild(body);
    return card;
}
