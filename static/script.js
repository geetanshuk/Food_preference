function choose_photo() {

    const chooseFile = confirm("OK = Choose from Files\nCancel = Take Photo");

    if (chooseFile) {
        // Trigger the file picker
        document.getElementById("fileInput").click();
    } else {
        // Trigger the camera
        document.getElementById("fileInput").click();
    }

}

// When user chooses a file, automatically send it
document.getElementById("fileInput").addEventListener("change", function () {
    const fileInput = this;

    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.text())
    .then(text => {
        document.getElementById("result").innerText = text;
    })
    .catch(err => {
        document.getElementById("result").innerText = "Error: " + err;
    });
});
