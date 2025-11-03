const form = document.getElementById("claimForm");
const input = document.getElementById("claimInput");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const sidebar = document.getElementById("sidebar");
const questionList = document.getElementById("questionList");

let currentData = null;

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const claim = input.value.trim();
    if (!claim) return;

    // Reset UI
    result.innerHTML = "";
    sidebar.classList.add("hidden");
    result.classList.add("hidden");
    loading.classList.remove("hidden");
    questionList.innerHTML = "";
    document.getElementById("summary").innerHTML = "";

    // Cria conexão SSE
    const eventSource = new EventSource(
        "http://127.0.0.1:8000/analyze-stream?" + new URLSearchParams({ claim })
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
            renderQuestions(finalResult);
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

    // Label
    const labelEl = document.createElement("span");
    labelEl.classList.add("label-badge");
    if (data.label) {
        labelEl.classList.add(
            data.label.toUpperCase() === "SUPPORTED" ? "label-supported" :
            data.label.toUpperCase() === "REFUTED" ? "label-refuted" :
            "label-uncertain"
        );
        labelEl.textContent = data.label.toUpperCase();
    } else {
        labelEl.classList.add("label-uncertain");
        labelEl.textContent = "Sem evidências suficientes";
    }
    result.appendChild(labelEl);

    // Confiança
    if (data.confidence !== null && data.confidence !== undefined) {
        const confEl = document.createElement("div");
        confEl.classList.add("confidence");
        confEl.style.marginLeft = "5px";
        confEl.textContent = `Confiança: ${data.confidence}%`;
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

// ====================================
// 🔹 Renderiza perguntas e evidências
// ====================================
function renderQuestions(data) {
    const questions = data.questions?.questions || [];
    questionList.innerHTML = "";

    if (questions.length === 0) {
        questionList.innerHTML = "<li>Nenhuma pergunta gerada.</li>";
        return;
    }

    questions.forEach((q, idx) => {
        const li = document.createElement("li");
        li.textContent = `${q} (${getEvidenceCount(q)} evidências)`;
        if (idx === 0) li.classList.add("active");
        li.addEventListener("click", () => {
            document.querySelectorAll("#questionList li").forEach(el => el.classList.remove("active"));
            li.classList.add("active");
            renderEvidence(q);
        });
        questionList.appendChild(li);
    });

    // Exibe a primeira evidência automaticamente
    renderEvidence(questions[0]);
    renderSummary();
}

// ===============================
// 🔹 Conta evidências por pergunta
// ===============================
function getEvidenceCount(question) {
    const evidenceObj = currentData?.evidences?.find(ev => ev.question === question);
    return evidenceObj ? evidenceObj.results.length : 0;
}

// ===============================
// 🔹 Renderiza o resumo lateral
// ===============================
function renderSummary() {
    const summaryDiv = document.getElementById("summary");
    summaryDiv.innerHTML = "<h3>Resumo de Evidências</h3>";
    const questions = currentData?.questions?.questions || [];

    if (questions.length === 0) return;

    const ul = document.createElement("ul");
    questions.forEach(q => {
        const li = document.createElement("li");
        li.textContent = `${q} → ${getEvidenceCount(q)} evidências`;
        ul.appendChild(li);
    });
    summaryDiv.appendChild(ul);
}

// ========================================
// 🔹 Renderiza evidências no painel central
// ========================================
function renderEvidence(question) {
    // Remove evidências antigas
    result.querySelectorAll(".evidence-card").forEach(c => c.remove());

    const evidenceObj = currentData?.evidences?.find(ev => ev.question === question);
    if (evidenceObj && evidenceObj.results.length > 0) {
        const card = document.createElement("div");
        card.classList.add("evidence-card");

        const title = document.createElement("h3");
        title.textContent = question;
        card.appendChild(title);

        const ul = document.createElement("ul");
        evidenceObj.results.forEach(res => {
            const li = document.createElement("li");
            li.innerHTML = `<a href="${res.link}" target="_blank">${res.title}</a><p>${res.snippet}</p>`;
            ul.appendChild(li);
        });

        card.appendChild(ul);
        result.appendChild(card);
    } else {
        const msg = document.createElement("p");
        msg.textContent = "Nenhuma evidência encontrada para esta pergunta.";
        result.appendChild(msg);
    }
}
