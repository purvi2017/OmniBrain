async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("status");

    if (fileInput.files.length === 0) {
        status.innerText = "Please select a PDF file";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/upload",
            {
                method: "POST",
                body: formData
            }
        );

        console.log("Status:", response.status);

const text = await response.text();

console.log("Response:", text);

if (!response.ok) {
    throw new Error(text);
}

status.innerText = "Upload Successful ✅";


    } catch (error) {

        console.error(error);
        status.innerText = "Upload Failed ❌";
    }
}