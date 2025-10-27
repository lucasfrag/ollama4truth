const form = document.getElementById("claimForm");
const input = document.getElementById("claimInput");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const claim = input.value.trim();
    if (!claim) return;

    result.classList.add("hidden");
    loading.classList.remove("hidden");

    try {
        const response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ claim })
        });

        const data = await response.json();
        loading.classList.add("hidden");
        result.classList.remove("hidden");
        result.textContent = JSON.stringify(data, null, 4);
    } catch (error) {
        loading.classList.add("hidden");
        result.classList.remove("hidden");
        result.textContent = "❌ Erro ao conectar com a API: " + error;
    }
});
