// Funkcja do wyświetlenia obrazu, jeśli src nie jest puste
function showImageIfSrcNotEmpty(imgElement, imgSrc) {
    if (imgSrc) {
        imgElement.src = imgSrc;
        imgElement.style.display = 'block';  // Pokaż obraz
    } else {
        imgElement.style.display = 'none';   // Ukryj obraz, jeśli src jest puste
    }
} 



var slider = document.getElementById("myRange");
var output = document.getElementById("num_samples");
output.innerHTML = slider.value;

slider.oninput = function() {
  output.innerHTML = this.value;
}

// Pobranie wartości ze slidera i wywołanie odpowiedniej funkcji
document.getElementById('quantum-classifier-btn').onclick = function() {
    document.getElementById('quantum-execution-time').textContent = "Running...";

    // Pobierz wartość ze slidera
    const numSamples = document.getElementById('myRange').value;

    // Wysłanie zapytania z num_samples do backendu
    fetch('/train_quantum', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ num_samples: numSamples })
    })
    .then(response => {
        return response.blob().then(imageBlob => {
            const imageObjectURL = URL.createObjectURL(imageBlob);
            const quantumImgElement = document.getElementById('quantum-classifier-img');
            showImageIfSrcNotEmpty(quantumImgElement, imageObjectURL);

            // Odbierz czas wykonania z nagłówków odpowiedzi
            const executionTime = response.headers.get('Execution-Time');
            document.getElementById('quantum-execution-time').innerText = `Execution time: ${executionTime} seconds`;
        });
    })
    .catch(error => console.error('Błąd:', error));
};



// Pobranie wartości ze slidera i wywołanie odpowiedniej funkcji
document.getElementById('classical-classifier-btn').onclick = function() {
    document.getElementById('classical-execution-time').textContent = "Running...";

    // Pobierz wartość ze slidera
    const numSamples = document.getElementById('myRange').value;

    // Wysłanie zapytania z num_samples do backendu
    fetch('/train_classical', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ num_samples: numSamples })
    })
    .then(response => {
        return response.blob().then(imageBlob => {
            const imageObjectURL = URL.createObjectURL(imageBlob);
            const classicalImgElement = document.getElementById('classical-classifier-img');
            showImageIfSrcNotEmpty(classicalImgElement, imageObjectURL);

            // Odbierz czas wykonania z nagłówków odpowiedzi
            const executionTime = response.headers.get('Execution-Time');
            document.getElementById('classical-execution-time').innerText = `Execution time: ${executionTime} seconds`;
        });
    })
    .catch(error => console.error('Błąd:', error));
};
