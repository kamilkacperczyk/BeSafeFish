// ===== Czasteczki w hero =====
(function() {
    var container = document.getElementById('particles');
    if (!container) return;

    for (var i = 0; i < 30; i++) {
        var particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = (60 + Math.random() * 40) + '%';
        particle.style.animationDuration = (6 + Math.random() * 10) + 's';
        particle.style.animationDelay = (-Math.random() * 10) + 's';
        particle.style.width = (2 + Math.random() * 2) + 'px';
        particle.style.height = particle.style.width;

        var colors = [
            'rgba(59, 130, 246, 0.5)',
            'rgba(6, 182, 212, 0.4)',
            'rgba(139, 92, 246, 0.3)'
        ];
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];

        container.appendChild(particle);
    }
})();

// ===== Formularz rejestracji =====
document.getElementById('registerForm').addEventListener('submit', function(e) {
    e.preventDefault();

    var username = document.getElementById('reg-username').value.trim();
    var email = document.getElementById('reg-email').value.trim();
    var password = document.getElementById('reg-password').value;
    var confirm = document.getElementById('reg-confirm').value;
    var terms = document.getElementById('reg-terms').checked;
    var msg = document.getElementById('formMessage');

    msg.textContent = '';
    msg.className = 'form-message';

    if (username.length < 3) {
        msg.textContent = 'Nazwa użytkownika musi mieć min. 3 znaki.';
        msg.classList.add('error');
        return;
    }

    if (!email) {
        msg.textContent = 'Podaj adres email.';
        msg.classList.add('error');
        return;
    }

    if (password.length < 8) {
        msg.textContent = 'Hasło musi mieć min. 8 znaków.';
        msg.classList.add('error');
        return;
    }

    if (password.length > 64) {
        msg.textContent = 'Hasło może mieć maks. 64 znaki.';
        msg.classList.add('error');
        return;
    }

    if (password !== confirm) {
        msg.textContent = 'Hasła się nie zgadzają.';
        msg.classList.add('error');
        return;
    }

    if (!terms) {
        msg.textContent = 'Musisz zaakceptować regulamin serwisu.';
        msg.classList.add('error');
        return;
    }

    // Wyslij do backendu
    var submitBtn = this.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Tworzenie konta...';

    var form = this;
    fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username, email: email, password: password })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        if (data.ok) {
            msg.textContent = data.msg;
            msg.classList.add('success');
            form.reset();
        } else {
            msg.textContent = data.msg;
            msg.classList.add('error');
        }
    })
    .catch(function() {
        msg.textContent = 'Błąd połączenia z serwerem.';
        msg.classList.add('error');
    })
    .finally(function() {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Stwórz konto';
    });
});

// ===== Smooth scroll dla linkow nawigacji =====
document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
        var target = document.querySelector(this.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ===== Navbar tlo przy scrollu =====
var navbar = document.querySelector('.navbar');
window.addEventListener('scroll', function() {
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(10, 14, 23, 0.95)';
    } else {
        navbar.style.background = 'rgba(10, 14, 23, 0.85)';
    }
});

// ===== Animacja kart przy scrollu =====
var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card, .step, .gallery-card, .plan-card, .download-card').forEach(function(el) {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
});
