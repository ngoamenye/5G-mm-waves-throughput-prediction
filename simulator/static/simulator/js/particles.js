const canvas = document.getElementById("bg");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let particles = [];

for (let i = 0; i < 100; i++) {
    particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 2 + 1,
        d: Math.random() * 2,
    });
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "cyan";
    for (let i = 0; i < particles.length; i++) {
        let p = particles[i];
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2, true);
        ctx.fill();
    }

    move();
}

function move() {
    for (let i = 0; i < particles.length; i++) {
        let p = particles[i];
        p.y += p.d;
        if (p.y > canvas.height) {
            p.y = 0;
            p.x = Math.random() * canvas.width;
        }
    }
}

setInterval(draw, 33);
