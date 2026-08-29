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
    formData.append("file", file);

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/upload",
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

    } catch (error) {

        console.error("Upload Error:", error);

        status.innerText =
            "Unable to connect to backend ❌";
    }
}


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

        /* ---------- API Request ---------- */

        const response = await fetch(
            "http://127.0.0.1:8000/query",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    query: query
                })
            }
        );


        console.log(
            "Query Status:",
            response.status
        );


        /* ---------- Read Response ---------- */

        const data = await response.json();

        console.log(
            "Query Response:",
            data
        );


        /* ---------- API Error ---------- */

        if (!response.ok) {

            throw new Error(
                data.detail || "Query failed"
            );
        }


        /* ---------- Display AI Response ---------- */

        answer.innerText =
            data.answer ||
            "No answer available";


        sourceDocument.innerText =
            data.source_document ||
            "N/A";


        documentId.innerText =
            data.document_id ||
            "N/A";


        relevantContext.innerText =
            data.relevant_context ||
            "No relevant context found";


        /* ---------- Success ---------- */

        queryStatus.innerText =
            "Query completed successfully ✓";
        queryStatus.classList.remove("processing");

document.getElementById("answerBox")
    .classList.add("answer-ready");

        /* ---------- Scroll to Answer ---------- */

        document.getElementById("answerBox")
            .scrollIntoView({
                behavior: "smooth",
                block: "start"
            });


    } catch (error) {

        console.error(
            "Query Error:",
            error
        );


        /* ---------- Error Display ---------- */
        queryStatus.classList.remove("processing");

        queryStatus.innerText =
            "Query failed. Please try again ❌";


        answer.innerText =
            "Unable to get an answer from OmniBrain.";


        sourceDocument.innerText =
            "N/A";


        documentId.innerText =
            "N/A";


        relevantContext.innerText =
            "Please make sure the backend server is running and a document has been uploaded.";


    } finally {

        /* ---------- Restore Button ---------- */

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