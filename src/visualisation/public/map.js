var map = L.map('map').setView([46.55, 15.65], 8); 

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
}).addTo(map);

var river = { name: "Drava", lat: 46.55, lon: 15.65 };

function onMarkerClick(marker, riverName) {
    $.ajax({
        url: 'https://api.open-meteo.com/v1/forecast?latitude=46.5547&longitude=15.6467&current=temperature_2m,rain,weather_code&timezone=Europe%2FBerlin&forecast_days=1', 
        type: 'GET',
        success: function(apiData) {
            var temperature = apiData.current.temperature_2m;
            var rain = apiData.current.rain;
            var weatherCode = apiData.current.weather_code;

            var requestData = {
                "Temperature 2m": temperature,
                "Rain": rain,
                "Weather Code": weatherCode,
                "Merilno Mesto": riverName,
                "Pretok Znacilni": "srednji pretok"
            };

            $.ajax({
                url: 'http://localhost:5000/predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(requestData),
                success: function(data) {
                    var prediction = data.prediction[0][0];
                    var flowDescription;
                    if (prediction <= 0.5) {
                        flowDescription = "Small";
                    } else if (prediction > 0.5 && prediction <= 1.5) {
                        flowDescription = "Medium";
                    } else {
                        flowDescription = "Large";
                    }
                    
                    marker.setPopupContent(riverName + "<br>River Flow: " + flowDescription);
                    marker.openPopup();
                },
                error: function(error) {
                    console.error("Error fetching prediction data:", error);
                    marker.setPopupContent(riverName + "<br>Error fetching prediction data. Please try again later.");
                    marker.openPopup();
                }
            });
        },
        error: function(error) {
            console.error("Error fetching weather data:", error);
            marker.setPopupContent(riverName + "<br>Error fetching weather data. Please try again later.");
            marker.openPopup();
        }
    });
}

var marker = L.marker([river.lat, river.lon]).addTo(map);
marker.bindPopup(river.name);
marker.on('click', function() {
    onMarkerClick(marker, river.name);
});
