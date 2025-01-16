const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadedImage = document.getElementById('uploaded-image');
const resultImage = document.getElementById('result-image');

// Drag & Drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
});

function handleFileUpload(file) {
    if (file) {
    const reader = new FileReader();
    reader.onload = () => {
        uploadedImage.src = reader.result;

        // Send the image to the server
        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
        method: 'POST',
        body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            resultImage.src = url;
        })
        .catch(console.error);
    };
    reader.readAsDataURL(file);
    }
}