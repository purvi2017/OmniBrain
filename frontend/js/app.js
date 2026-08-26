async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("status");

    if (!fileInput.files.length) {
        status.innerText = "Please select a PDF file";
        return;
    }

    status.innerText = "Uploading...";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {

        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });

        console.log("Upload Status:", response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Upload Error:", errorText);
            status.innerText = "Upload Failed ❌";
            return;
        }

        status.innerText = "Upload Successful ✅";

        const result = await response.text();
        console.log("Upload Response:", result);

    } catch (error) {

        console.error("Upload Error:", error);
        status.innerText = "Upload Failed ❌";
    }
}

async function askQuery() {

    const queryInput = document.getElementById("queryInput");
    const queryStatus = document.getElementById("queryStatus");

    const answer = document.getElementById("answer");
    const sourceDocument = document.getElementById("sourceDocument");
    const documentId = document.getElementById("documentId");
    const relevantContext = document.getElementById("relevantContext");

    const query = queryInput.value.trim();

    if (!query) {
        queryStatus.innerText = "Please enter a question";
        return;
    }

    queryStatus.innerText = "Processing...";

    try {

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

        console.log("Status:", response.status);

        const data = await response.json();

        console.log("Response:", data);

        if (!response.ok) {
            throw new Error(data.detail || "Query failed");
        }

        answer.innerText = data.answer || "No answer available";
        sourceDocument.innerText = data.source_document || "N/A";
        documentId.innerText = data.document_id || "N/A";
        relevantContext.innerText = data.relevant_context || "No relevant context found";

        queryStatus.innerText = "Query completed successfully ✅";

    } catch (error) {

        console.error(error);

        queryStatus.innerText = "Query failed ❌";
        answer.innerText = "";
        sourceDocument.innerText = "";
        documentId.innerText = "";
        relevantContext.innerText = "";
    }
}