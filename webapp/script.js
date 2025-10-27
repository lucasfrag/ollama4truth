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

    result.innerHTML = "";
    sidebar.classList.add("hidden");
    result.classList.add("hidden");
    loading.classList.remove("hidden");
    questionList.innerHTML = "";
    document.getElementById("summary").innerHTML = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ claim })
        });

        const data = await response.json();
        currentData = data;

        loading.classList.add("hidden");
        sidebar.classList.remove("hidden");
        result.classList.remove("hidden");

        renderClaim(data);
        renderQuestions(data);

    } catch (error) {
        loading.classList.add("hidden");
        result.classList.remove("hidden");
        result.textContent = "❌ Erro ao conectar com a API: " + error;
    }
});

// Renderiza a claim e badge de status
function renderClaim(data) {
    result.innerHTML = "";

    const claimEl = document.createElement("div");
    claimEl.classList.add("claim-title");
    claimEl.textContent = "📝 " + data.claim;
    result.appendChild(claimEl);

    if (data.label) {
        const labelEl = document.createElement("span");
        labelEl.classList.add("label-badge");
        labelEl.classList.add(
            data.label.toLowerCase() === "supported" ? "label-supported" :
            data.label.toLowerCase() === "refuted" ? "label-refuted" :
            "label-uncertain"
        );
        labelEl.textContent = data.label;
        result.appendChild(labelEl);
    } else {
        const labelEl = document.createElement("span");
        labelEl.classList.add("label-badge", "label-uncertain"); // usa classe "incerto"
        labelEl.textContent = "Sem evidências suficientes";
        result.appendChild(labelEl);
    }
}

// Renderiza a lista de perguntas na sidebar
function renderQuestions(data) {
    const questions = data.questions.questions;
    questionList.innerHTML = "";

    questions.forEach((q, idx) => {
        const li = document.createElement("li");
        li.textContent = `${q} (${getEvidenceCount(q)} evidências)`;
        if(idx === 0) li.classList.add("active");
        li.addEventListener("click", () => {
            document.querySelectorAll("#questionList li").forEach(el => el.classList.remove("active"));
            li.classList.add("active");
            renderEvidence(q);
        });
        questionList.appendChild(li);
    });

    renderEvidence(questions[0]);
    renderSummary();
}

// Conta quantas evidências existem para cada pergunta
function getEvidenceCount(question) {
    const evidenceObj = currentData.evidences.find(ev => ev.question === question);
    return evidenceObj ? evidenceObj.results.length : 0;
}

// Renderiza resumo de evidências na sidebar
function renderSummary() {
    const summaryDiv = document.getElementById("summary");
    summaryDiv.innerHTML = "";
    /*
    currentData.questions.questions.forEach(q => {
        const span = document.createElement("span");
        span.textContent = `${q}: ${getEvidenceCount(q)} evidências`;
        span.style.display = "block";
        span.style.marginBottom = "5px";
        summaryDiv.appendChild(span);
    });*/
}

// Renderiza evidências no painel principal
function renderEvidence(question) {
    // Remove cards antigos
    result.querySelectorAll(".evidence-card").forEach(c => c.remove());

    const evidenceObj = currentData.evidences.find(ev => ev.question === question);
    if(evidenceObj && evidenceObj.results.length > 0) {
        const card = document.createElement("div");
        card.classList.add("evidence-card");

        const title = document.createElement("h3");
        title.textContent = question;
        card.appendChild(title);

        const ul = document.createElement("ul");
        evidenceObj.results.forEach(res => {
            const li = document.createElement("li");
            li.innerHTML = `<a href="${res.link}" target="_blank">${res.title}</a>: ${res.snippet}`;
            ul.appendChild(li);
        });

        card.appendChild(ul);
        result.appendChild(card);
    }
}
