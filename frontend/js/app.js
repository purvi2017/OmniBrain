/* =====================================================
   OMNIBRAIN - UPLOAD PDF
   ===================================================== */

async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("status");

    if (!fileInput.files.length) {
        status.innerText = "Please select a PDF file";
        return;
    }

    const file = fileInput.files[0];

    // Basic PDF validation
    if (file.type !== "application/pdf") {
        status.innerText = "Please select a valid PDF file ❌";
        return;
    }

    status.innerText = "Uploading document...";

    const formData = new FormData();
    formData.append("files", file);

    try {

        const response = await fetch(
            "http://127.0.0.1:8001/ingest",
            {
                method: "POST",
                body: formData
            }
        );

        console.log("Upload Status:", response.status);

        if (!response.ok) {

            const errorText = await response.text();

            console.error("Upload Error:", errorText);

            status.innerText = "Upload Failed ❌";
            return;
        }

        const result = await response.text();

        console.log("Upload Response:", result);

        status.innerText = "Document uploaded successfully ✅";
        setTimeout(() => {
            window.location.href = "../query.html";
        }, 800);
    } catch (error) {

        console.error("Upload Error:", error);

        status.innerText =
            "Unable to connect to backend ❌";
    }
}


/* =====================================================
   OMNIBRAIN - QUERY
   ===================================================== */

/* =====================================================
   OMNIBRAIN - QUERY
   ===================================================== */

async function askQuery() {

    const queryInput =
        document.getElementById("queryInput");

    const queryStatus =
        document.getElementById("queryStatus");

    const answer =
        document.getElementById("answer");

    const sourceDocument =
        document.getElementById("sourceDocument");

    const documentId =
        document.getElementById("documentId");

    const relevantContext =
        document.getElementById("relevantContext");

    const askButton =
        document.querySelector(".query-box button");

    const query =
        queryInput.value.trim();


    /* ---------- Validation ---------- */

    if (!query) {

        queryStatus.innerText =
            "Please enter a question";

        queryInput.focus();

        return;
    }


    /* ---------- Session ID ---------- */

    let sessionId =
        sessionStorage.getItem("omnibrain_session_id");

    if (!sessionId) {

        sessionId =
            "session-" +
            Date.now() +
            "-" +
            Math.random()
                .toString(36)
                .substring(2, 8);

        sessionStorage.setItem(
            "omnibrain_session_id",
            sessionId
        );
    }


    /* ---------- Loading State ---------- */

    queryStatus.innerText =
        "OmniBrain is thinking...";

    queryStatus.classList.add("processing");

    if (askButton) {
        askButton.disabled = true;
        askButton.style.opacity = "0.7";
        askButton.style.cursor = "wait";
    }


    try {

        /* ---------- CHAT API Request ---------- */

        const response = await fetch(
            `http://127.0.0.1:8001/chat/${sessionId}`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    message: query,
                    top_k: 3,
                    min_score: 0.35,
                    max_score_drop: 0.25,
                    document_id: null
                })
            }
        );


        console.log(
            "Chat Status:",
            response.status
        );


        /* ---------- Read Response ---------- */

        const data = await response.json();

        console.log(
            "Chat Response:",
            data
        );


        /* ---------- API Error ---------- */

        if (!response.ok) {

            throw new Error(
                data.detail || "Chat request failed"
            );
        }


        /* ---------- Display Answer ---------- */

        if (answer) {
            answer.innerText =
                data.answer || "No answer received.";
        }


        /* ---------- Source Document ---------- */

        if (sourceDocument) {
            sourceDocument.innerText =
                data.source_document || "N/A";
        }


        /* ---------- Document ID ---------- */

        if (documentId) {
            documentId.innerText =
                data.document_id || "N/A";
        }


        /* ---------- Relevant Context ---------- */

        if (relevantContext) {
            relevantContext.innerText =
                data.relevant_context ||
                "No relevant context found";
        }


        /* ---------- Success ---------- */

        queryStatus.innerText =
            "Query completed successfully ✓";

        queryStatus.classList.remove("processing");


    } catch (error) {

        console.error(
            "Chat Error:",
            error
        );

        queryStatus.innerText =
            "Unable to get answer ❌";

        queryStatus.classList.remove("processing");

        if (answer) {
            answer.innerText =
                error.message ||
                "Something went wrong.";
        }

    } finally {

        if (askButton) {
            askButton.disabled = false;
            askButton.style.opacity = "1";
            askButton.style.cursor = "pointer";
        }
    }
}


/* =====================================================
   ENTER KEY SUPPORT
   ===================================================== */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const queryInput =
            document.getElementById("queryInput");

        if (queryInput) {

            queryInput.addEventListener(
                "keydown",
                function (event) {

                    if (
                        event.key === "Enter"
                    ) {

                        event.preventDefault();

                        askQuery();
                    }
                }
            );
        }

    }
);
function setQuery(text) {
    const queryInput = document.getElementById("queryInput");

    if (queryInput) {
        queryInput.value = text;
        queryInput.focus();
    }
}