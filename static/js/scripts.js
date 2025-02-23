function updateTime() {
    const currentTimeElement = document.getElementById("currentTime");
    const now = new Date();
    // Format time
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    const timeString = `${hours}:${minutes}:${seconds}`;
    // Format date
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const dateString = now.toLocaleDateString(undefined, options);
    // Display date and time
    currentTimeElement.innerHTML = `<span class="date">${dateString}</span> <span class="time">${timeString}</span>`;
}

setInterval(updateTime, 1000);
updateTime();
document.getElementById('feedbackForm').addEventListener('submit', function(event) {
    event.preventDefault(); 
    const name = document.getElementById('name').value;
    const feedback = document.getElementById('feedback').value;
    showMessage(`Thank you, ${name}! Your feedback has been submitted.`, 'success');
    this.reset();
});

function showMessage(message, type) {
    const messageDiv = document.getElementById('message');
    messageDiv.textContent = message;
    messageDiv.className = type; 
    messageDiv.classList.remove('hidden'); 
    setTimeout(() => {
        messageDiv.classList.add('hidden');
    }, 3000);
}

var lat = 40.7128; // Example latitude
var lon = -74.0060; // Example longitude
var city = "New York"; // Example city
var date = "2023-10-01"; // Example date
var predicted_crime = "low"; // Example predicted crime

var map = L.map('map').setView([lat, lon], 10);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);
L.marker([lat, lon]).addTo(map)
    .bindPopup(`Predicted crime in ${city} on ${date} is ${predicted_crime}`)
    .openPopup();