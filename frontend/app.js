// Wait for the DOM to be fully loaded before running the script
document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-file');
    const processButton = document.getElementById('processButton');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const resultsOutput = document.getElementById('results-output');
    const imagePreview = document.getElementById('image-preview');

    imageInput.value = null; // Clears the file input field
    
    imagePreview.style.display = 'none';
    imagePreview.src = '';
    
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); 

        const file = imageInput.files[0];

        if (!file) {
            resultsOutput.textContent = "Please select an image file first.";
            resultsSection.style.display = 'block';
            return;
        }

        setLoading(true);

        const formData = new FormData();
        formData.append('image', file);

        const backendUrl = 'http://localhost:5000/process_image';

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json(); // data = { "matched_img_index": 123, "image_b64": "..." }

            if (response.ok) {
                const resultImage = document.getElementById('result-image');
                const imageSrc = "data:image/png;base64," + data.image_b64;
                resultImage.src = imageSrc;
                resultImage.style.display = 'block';

                const index = data.matched_img_index;
                const outputText = `Nearest index in training set: ${index}`

                resultsOutput.textContent = outputText;
            
            } else {
                document.getElementById('result-image').style.display = 'none';
                throw new Error(data.error || 'Unknown server error');
            }

        } catch (error) {
            console.error('Error:', error);
            resultsOutput.textContent = `Error connecting to backend:\n${error.message}\n\nIs the Python server running?`;
        } finally {
            setLoading(false);
            resultsSection.classList.remove('hidden');
        }
    });

    imageInput.addEventListener('change', async () => {
        const file = imageInput.files[0];
        const imagePreview = document.getElementById('image-preview');

        if (file) {
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('http://localhost:5000/preview', {
                    method: 'POST',
                    body: formData,
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const imageSrc = "data:image/png;base64," + data.image_b64;
                    imagePreview.src = imageSrc;
                    imagePreview.style.display = 'block';
                } else {
                    throw new Error(data.error);
                }

            } catch (error) {
                console.error("Preview failed:", error);
                imagePreview.src = '';
                imagePreview.style.display = 'none';
            }

        } else {
            imagePreview.src = '';
            imagePreview.style.display = 'none';
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            processButton.disabled = true;
            processButton.textContent = 'Processing...';
            
            loader.classList.remove('hidden');
            resultsSection.classList.add('hidden'); 
            resultsOutput.textContent = '';
        } else {
            processButton.disabled = false;
            processButton.textContent = 'Process Image';
            
            loader.classList.add('hidden');
        }
    }
});