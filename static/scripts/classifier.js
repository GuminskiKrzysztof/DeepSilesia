function showImageIfSrcNotEmpty(imgElement, imgSrc) {
    if (imgSrc) {
        imgElement.src = imgSrc;
        imgElement.style.display = 'block';
    } else {
        imgElement.style.display = 'none';
    }
} 



var slider = document.getElementById("myRange");
var output = document.getElementById("num_samples");
output.innerHTML = slider.value;

slider.oninput = function() {
  output.innerHTML = this.value;
}

document.getElementById('quantum-classifier-btn').onclick = function() {
    document.getElementById('quantum-execution-time').textContent = "Running...";

    const numSamples = document.getElementById('myRange').value;

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

            const executionTime = response.headers.get('Execution-Time');
            document.getElementById('quantum-execution-time').innerText = `Execution time: ${executionTime} seconds`;
        });
    })
    .catch(error => console.error('Błąd:', error));
};


document.getElementById('classical-classifier-btn').onclick = function() {
    document.getElementById('classical-execution-time').textContent = "Running...";

    const numSamples = document.getElementById('myRange').value;

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

            const executionTime = response.headers.get('Execution-Time');
            document.getElementById('classical-execution-time').innerText = `Execution time: ${executionTime} seconds`;
        });
    })
    .catch(error => console.error('Błąd:', error));
};
