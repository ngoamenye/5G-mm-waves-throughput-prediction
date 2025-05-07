// Extrait simplifié - à compléter avec vos données dynamiques
let map;
function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 33.89, lng: 10.1 },
    zoom: 15,
  });

  const antenna1 = { lat: 33.891, lng: 10.102 };
  const antenna2 = { lat: 33.893, lng: 10.108 };

  new google.maps.Marker({ position: antenna1, map, label: "A1" });
  new google.maps.Marker({ position: antenna2, map, label: "A2" });

  const user = new google.maps.Marker({
    position: antenna1,
    map,
    icon: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png'
  });

  let step = 0;
  const steps = 50;
  const interval = setInterval(() => {
    const lat = antenna1.lat + (antenna2.lat - antenna1.lat) * (step / steps);
    const lng = antenna1.lng + (antenna2.lng - antenna1.lng) * (step / steps);
    user.setPosition({ lat, lng });
    step++;
    if (step > steps) clearInterval(interval);
  }, 200);
}

window.onload = initMap;
