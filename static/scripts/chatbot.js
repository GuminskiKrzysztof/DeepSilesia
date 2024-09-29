function toggleText() {
    var bubbleMessage = document.getElementById("bubbleMessage");
    if (bubbleMessage.style.display === "none" || bubbleMessage.style.display === "") {
        bubbleMessage.style.display = "block"; // Show 
    } else {
        bubbleMessage.style.display = "none"; // Hide 
    }
}