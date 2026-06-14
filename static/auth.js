/* ═══════════════════════════════════════════════════════════
   auth.js  —  TrafficCommand Auth Portal Logic
═══════════════════════════════════════════════════════════ */

'use strict';

// ── Particle system ───────────────────────────────────────
(function spawnParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  const COUNT = 20;
  for (let i = 0; i < COUNT; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left     = `${Math.random() * 100}%`;
    p.style.animationDuration = `${6 + Math.random() * 10}s`;
    p.style.animationDelay   = `${Math.random() * 8}s`;
    p.style.opacity  = (0.2 + Math.random() * 0.5).toString();
    p.style.width    = p.style.height = `${1 + Math.random() * 3}px`;
    container.appendChild(p);
  }
})();

// ── Traffic light animation ───────────────────────────────
(function cycleLights() {
  const lights  = document.querySelectorAll('.light');
  if (!lights.length) return;
  let current   = 2; // start green
  setInterval(() => {
    lights.forEach(l => l.classList.remove('active'));
    current = (current + 1) % 3;
    lights[current].classList.add('active');
  }, 1800);
})();

// ── Toggle password visibility ────────────────────────────
function togglePw(inputId, btn) {
  const input = document.getElementById(inputId);
  if (!input) return;
  const isText = input.type === 'text';
  input.type = isText ? 'password' : 'text';
  btn.querySelector('svg').style.opacity = isText ? '1' : '0.5';
}

// ── Password strength meter ───────────────────────────────
function checkStrength(pw) {
  const fill  = document.getElementById('strengthFill');
  const label = document.getElementById('strengthLabel');
  if (!fill || !label) return;
  
  if (!pw) { fill.style.width = '0%'; label.textContent = ''; return; }

  let score = 0;
  if (pw.length >= 8)               score++;
  if (pw.length >= 12)              score++;
  if (/[A-Z]/.test(pw))             score++;
  if (/[0-9]/.test(pw))             score++;
  if (/[^A-Za-z0-9]/.test(pw))     score++;

  const levels = [
    { pct: '20%', color: '#ff4757', text: 'Weak',        textColor: '#ff4757' },
    { pct: '40%', color: '#ffb800', text: 'Fair',        textColor: '#ffb800' },
    { pct: '60%', color: '#ffb800', text: 'Good',        textColor: '#ffb800' },
    { pct: '80%', color: '#00d4ff', text: 'Strong',      textColor: '#00d4ff' },
    { pct: '100%',color: '#00ff88', text: 'Excellent',   textColor: '#00ff88' },
  ];
  const lvl = levels[Math.max(0, score - 1)];
  fill.style.width      = lvl.pct;
  fill.style.background = lvl.color;
  label.textContent     = lvl.text;
  label.style.color     = lvl.textColor;
}

// ── Field validation helpers ──────────────────────────────
function setError(inputId, errId, msg) {
  const input = document.getElementById(inputId);
  const err   = document.getElementById(errId);
  if (input)  { input.classList.add('error'); input.classList.remove('valid'); }
  if (err)    { err.textContent = msg; }
  return false;
}

function setValid(inputId, errId) {
  const input = document.getElementById(inputId);
  const err   = document.getElementById(errId);
  if (input)  { input.classList.remove('error'); input.classList.add('valid'); }
  if (err)    { err.textContent = ''; }
  return true;
}

function clearField(inputId, errId) {
  const input = document.getElementById(inputId);
  const err   = document.getElementById(errId);
  if (input)  { input.classList.remove('error', 'valid'); }
  if (err)    { err.textContent = ''; }
}

// ── Show alert ────────────────────────────────────────────
function showAlert(id, msg, type /* 'success' | 'error' */) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent  = msg;
  el.className    = `alert ${type}`;
  el.hidden       = false;
}

// ── Loading state ─────────────────────────────────────────
function setLoading(btnId, loading) {
  const btn     = document.getElementById(btnId);
  if (!btn) return;
  
  const textEl  = btn.querySelector('.btn-text') || btn;
  btn.disabled  = loading;
  
  if (loading) {
    btn.dataset.origText = textEl.textContent;
    textEl.textContent = "Processing...";
  } else {
    if (btn.dataset.origText) {
      textEl.textContent = btn.dataset.origText;
    }
  }
}

// ── Forgot password (placeholder) ────────────────────────
function showForgot() {
  alert('Please contact your system administrator to reset your password.');
}

/* ══════════════════════════════════════════════════════════
   LOGIN
══════════════════════════════════════════════════════════ */

async function handleLogin(e) {
  e.preventDefault();

  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  const isAdminEl = document.getElementById('login-is-admin');
  const is_admin = isAdminEl ? isAdminEl.checked : false;
  let   valid    = true;

  // Clear old states
  clearField('login-username', 'login-username-err');
  clearField('login-password', 'login-password-err');

  // Validate
  if (!username) {
    valid = setError('login-username', 'login-username-err', 'Username is required.') && valid;
  } else {
    setValid('login-username', 'login-username-err');
  }
  if (!password) {
    valid = setError('login-password', 'login-password-err', 'Password is required.') && valid;
  } else {
    setValid('login-password', 'login-password-err');
  }
  if (!valid) return;

  setLoading('loginBtn', true);

  try {
    const res  = await fetch('/api/auth/login', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ username, password, is_admin,
                                remember: document.getElementById('remember-me')?.checked }),
    });
    const data = await res.json();

    if (res.ok && data.success) {
      showAlert('login-alert', '✓ Login successful. Redirecting…', 'success');
      setTimeout(() => { window.location.href = data.redirect || '/'; }, 900);
    } else {
      showAlert('login-alert', data.error || 'Invalid credentials. Please try again.', 'error');
      setLoading('loginBtn', false);
    }
  } catch (err) {
    showAlert('login-alert', 'Network error. Please check your connection.', 'error');
    setLoading('loginBtn', false);
  }
}

/* ══════════════════════════════════════════════════════════
   SIGN UP
══════════════════════════════════════════════════════════ */

async function handleSignup(e) {
  e.preventDefault();

  const firstname = document.getElementById('signup-firstname').value.trim();
  const lastname  = document.getElementById('signup-lastname').value.trim();
  const badge     = document.getElementById('signup-badge').value.trim();
  const email     = document.getElementById('signup-email').value.trim();
  const username  = document.getElementById('signup-username').value.trim();
  const password  = document.getElementById('signup-password').value;
  let   valid     = true;

  // Clear all
  ['signup-firstname','signup-lastname','signup-badge',
   'signup-email','signup-username','signup-password']
    .forEach(id => clearField(id, id + '-err'));

  // Validate each field
  if (!firstname) {
    valid = setError('signup-firstname', 'signup-firstname-err', 'First name is required.') && valid;
  } else { setValid('signup-firstname', 'signup-firstname-err'); }

  if (!lastname) {
    valid = setError('signup-lastname', 'signup-lastname-err', 'Last name is required.') && valid;
  } else { setValid('signup-lastname', 'signup-lastname-err'); }

  if (!badge) {
    valid = setError('signup-badge', 'signup-badge-err', 'Badge / Officer ID is required.') && valid;
  } else { setValid('signup-badge', 'signup-badge-err'); }

  const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!email || !emailRe.test(email)) {
    valid = setError('signup-email', 'signup-email-err', 'Enter a valid email address.') && valid;
  } else { setValid('signup-email', 'signup-email-err'); }

  if (!username || username.length < 3) {
    valid = setError('signup-username', 'signup-username-err', 'Username must be at least 3 characters.') && valid;
  } else { setValid('signup-username', 'signup-username-err'); }

  if (!password) {
    valid = setError('signup-password', 'signup-password-err', 'Password is required.') && valid;
  } else { setValid('signup-password', 'signup-password-err'); }

  if (!valid) return;

  setLoading('signupBtn', true);

  try {
    const res  = await fetch('/api/auth/signup', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ firstname, lastname, badge_id: badge,
                                email, username, password }),
    });
    const data = await res.json();

    if (res.ok && data.success) {
      showAlert('signup-alert', '✓ Account created! Please wait for administrator approval.', 'success');
      
      // Clear form
      document.getElementById('signupForm').reset();
      const sFill = document.getElementById('strengthFill');
      const sLabel = document.getElementById('strengthLabel');
      if (sFill) sFill.style.width = '0%';
      if (sLabel) sLabel.textContent = '';
      
      setLoading('signupBtn', false);
    } else {
      showAlert('signup-alert', data.error || 'Registration failed. Please try again.', 'error');
      setLoading('signupBtn', false);
    }
  } catch (err) {
    showAlert('signup-alert', 'Network error. Please check your connection.', 'error');
    setLoading('signupBtn', false);
  }
}

// ── Live input validation (on blur) ──────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Login
  document.getElementById('login-username')?.addEventListener('blur', function () {
    if (this.value.trim()) setValid('login-username', 'login-username-err');
  });
  document.getElementById('login-password')?.addEventListener('blur', function () {
    if (this.value) setValid('login-password', 'login-password-err');
  });

  // Sign-up email
  document.getElementById('signup-email')?.addEventListener('blur', function () {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (this.value && re.test(this.value)) {
      setValid('signup-email', 'signup-email-err');
    } else if (this.value) {
      setError('signup-email', 'signup-email-err', 'Enter a valid email address.');
    }
  });

  // Password confirm
  document.getElementById('signup-confirm')?.addEventListener('input', function () {
    const pw = document.getElementById('signup-password').value;
    if (this.value && this.value !== pw) {
      setError('signup-confirm', 'signup-confirm-err', 'Passwords do not match.');
    } else if (this.value) {
      setValid('signup-confirm', 'signup-confirm-err');
    }
  });
});
