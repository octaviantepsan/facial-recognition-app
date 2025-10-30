// Wait for the DOM to be fully loaded before running the script
document.addEventListener("DOMContentLoaded", () => {

    // --- 1. Get elements using the CORRECT IDs from your HTML ---
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-file'); // Corrected ID
    const processButton = document.getElementById('processButton');
    const loader = document.getElementById('loader'); // Corrected ID
    const resultsSection = document.getElementById('results-section'); // Corrected ID
    const resultsOutput = document.getElementById('results-output'); // Corrected ID

    // --- 2. Listen for the FORM to be submitted ---
    // This is better than a button click, as it also works if the user hits "Enter"
    uploadForm.addEventListener('submit', async (event) => {
        // Stop the form from reloading the page
        event.preventDefault(); 

        const file = imageInput.files[0];

        // Check if a file was selected
        if (!file) {
            resultsOutput.textContent = "Please select an image file first.";
            resultsSection.style.display = 'block'; // Use style, not class (or use .classList.remove('hidden'))
            return;
        }

        // --- 3. Start the processing ---
        setLoading(true);

        const formData = new FormData();
        formData.append('image', file);

        const backendUrl = 'http://localhost:5000/process_image';

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                // Success! Show the results
                resultsOutput.textContent = JSON.stringify(data, null, 2); // Pretty-print JSON
            } else {
                // Error from the server
                throw new Error(data.error || 'Unknown server error');
            }

        } catch (error) {
            // Network error
            console.error('Error:', error);
            resultsOutput.textContent = `Error connecting to backend:\n${error.message}\n\nIs the Python server running?`;
        } finally {
            // --- 4. Stop loading and show results ---
            setLoading(false);
            resultsSection.classList.remove('hidden'); // Show the results section
        }
    });

    // --- 5. Helper function to show/hide loading state ---
    function setLoading(isLoading) {
        if (isLoading) {
            // Disable button and change text
            processButton.disabled = true;
            processButton.textContent = 'Processing...';
            
            // Show the spinner and hide old results
            loader.classList.remove('hidden');
            resultsSection.classList.add('hidden'); 
            resultsOutput.textContent = ''; // Clear old results
        } else {
            // Re-enable button and reset text
            processButton.disabled = false;
            processButton.textContent = 'Process Image';
            
            // Hide the spinner
            loader.classList.add('hidden');
        }
    }
});