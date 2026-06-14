'use strict';

document.addEventListener('DOMContentLoaded', () => {
  loadUsers();
});

async function loadUsers() {
  const tbody = document.getElementById('users-tbody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/auth/users');
    if (!res.ok) {
      if (res.status === 401 || res.status === 403) {
        window.location.href = '/login';
        return;
      }
      throw new Error('Failed to load users');
    }
    const users = await res.json();
    renderUsers(users);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--red);">Error loading users.</td></tr>`;
    showToast('Error loading users.', 'error');
  }
}

function renderUsers(users) {
  const tbody = document.getElementById('users-tbody');
  tbody.innerHTML = '';

  if (users.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center;">No users found.</td></tr>`;
    return;
  }

  users.forEach(user => {
    const tr = document.createElement('tr');
    
    // Status Badge
    const statusClass = user.approved ? 'approved' : 'pending';
    const statusText  = user.approved ? 'Approved' : 'Pending';
    
    // Role Badge
    const roleClass = user.role === 'admin' ? 'admin' : 'controller';
    
    // Format Date
    const date = new Date(user.created_at).toLocaleDateString();

    // Actions
    let actionsHtml = '';
    if (user.role !== 'admin') {
      if (user.approved) {
        actionsHtml += `<button class="btn-sm btn-revoke" onclick="revokeUser(${user.id})">Revoke</button>`;
      } else {
        actionsHtml += `<button class="btn-sm btn-approve" onclick="approveUser(${user.id})">Approve</button>`;
      }
      actionsHtml += `<button class="btn-sm btn-delete" onclick="deleteUser(${user.id})">Delete</button>`;
    } else {
      actionsHtml = `<span style="color: var(--text-muted); font-size: 0.8rem;">Superuser</span>`;
    }

    tr.innerHTML = `
      <td>${user.id}</td>
      <td>${escapeHtml(user.firstname)} ${escapeHtml(user.lastname)}</td>
      <td>${escapeHtml(user.badge_id)}</td>
      <td>${escapeHtml(user.username)}</td>
      <td>${escapeHtml(user.email)}</td>
      <td><span class="role-badge ${roleClass}">${user.role}</span></td>
      <td><span class="status-badge ${statusClass}">${statusText}</span></td>
      <td>${date}</td>
      <td class="action-buttons">${actionsHtml}</td>
    `;
    
    tbody.appendChild(tr);
  });
}

async function approveUser(id) {
  try {
    const res = await fetch(`/api/auth/users/${id}/approve`, { method: 'POST' });
    if (res.ok) {
      showToast('User approved successfully.', 'success');
      loadUsers();
    } else {
      showToast('Failed to approve user.', 'error');
    }
  } catch (err) {
    showToast('Network error.', 'error');
  }
}

async function revokeUser(id) {
  try {
    const res = await fetch(`/api/auth/users/${id}/revoke`, { method: 'POST' });
    if (res.ok) {
      showToast('User access revoked.', 'success');
      loadUsers();
    } else {
      showToast('Failed to revoke user.', 'error');
    }
  } catch (err) {
    showToast('Network error.', 'error');
  }
}

async function deleteUser(id) {
  if (!confirm('Are you sure you want to delete this user? This cannot be undone.')) return;
  try {
    const res = await fetch(`/api/auth/users/${id}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('User deleted.', 'success');
      loadUsers();
    } else {
      showToast('Failed to delete user.', 'error');
    }
  } catch (err) {
    showToast('Network error.', 'error');
  }
}

async function logout() {
  try {
    await fetch('/api/auth/logout', { method: 'POST' });
    window.location.href = '/login';
  } catch (err) {
    window.location.href = '/login';
  }
}

function showToast(msg, type) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
