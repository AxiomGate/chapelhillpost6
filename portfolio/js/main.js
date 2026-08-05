/* AxiomGate Labs — portfolio rendering logic (no framework, no build step) */

(function () {
  'use strict';

  const state = { category: 'all', query: '' };

  const gridEl = document.getElementById('project-grid');
  const pillsEl = document.getElementById('filter-pills');
  const statsEl = document.getElementById('stats-row');
  const searchEl = document.getElementById('search-input');
  const modalBackdrop = document.getElementById('modal-backdrop');
  const modalContent = document.getElementById('modal-content');
  const modalClose = document.getElementById('modal-close');

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str == null ? '' : String(str);
    return div.innerHTML;
  }

  function categoryOf(id) {
    return CATEGORIES.find((c) => c.id === id);
  }

  function repoCountOf(project) {
    return (project.repos ? project.repos.length : 0) + (project.variants ? project.variants.length : 0);
  }

  function totalRepoCount() {
    return PROJECTS.reduce((sum, p) => sum + repoCountOf(p), 0);
  }

  // ── Stats bar ─────────────────────────────────────────────
  function renderStats() {
    const liveCount = PROJECTS.filter((p) => p.status === 'live').length;
    const stats = [
      { num: PROJECTS.length, label: 'Projects' },
      { num: CATEGORIES.length, label: 'Categories' },
      { num: liveCount, label: 'Live right now' },
      { num: totalRepoCount(), label: 'Repositories' },
    ];
    statsEl.innerHTML = stats
      .map((s) => `<div><div class="stat-num">${s.num}</div><div class="stat-label">${s.label}</div></div>`)
      .join('');
  }

  // ── Filter pills ──────────────────────────────────────────
  function renderPills() {
    const counts = {};
    PROJECTS.forEach((p) => (counts[p.category] = (counts[p.category] || 0) + 1));

    const pills = [
      `<button class="pill ${state.category === 'all' ? 'active' : ''}" data-cat="all">All <span class="count">${PROJECTS.length}</span></button>`,
    ].concat(
      CATEGORIES.map(
        (c) =>
          `<button class="pill ${state.category === c.id ? 'active' : ''}" data-cat="${c.id}">` +
          `<span class="pill-dot" style="background:var(--cat-${c.id})"></span>${c.name} <span class="count">${counts[c.id] || 0}</span></button>`
      )
    );
    pillsEl.innerHTML = pills.join('');

    pillsEl.querySelectorAll('.pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.category = btn.dataset.cat;
        renderPills();
        renderGrid();
      });
    });
  }

  // ── Grid ──────────────────────────────────────────────────
  function matchesSearch(project, q) {
    if (!q) return true;
    const haystack = [
      project.name,
      project.summary,
      project.whatItIs,
      project.problem,
      ...(project.tech || []),
      categoryOf(project.category) ? categoryOf(project.category).name : '',
    ]
      .join(' ')
      .toLowerCase();
    return haystack.includes(q.toLowerCase());
  }

  function cardHtml(project) {
    const cat = categoryOf(project.category);
    const status = STATUSES[project.status] || { label: project.status };
    const techChips = (project.tech || [])
      .slice(0, 4)
      .map((t) => `<span class="tech-chip">${escapeHtml(t)}</span>`)
      .join('');
    const moreCount = (project.tech || []).length > 4 ? `<span class="card-more">+${project.tech.length - 4}</span>` : '';

    return `
      <article class="card" data-id="${project.id}" tabindex="0" role="button" aria-label="View ${escapeHtml(project.name)} details">
        <div class="card-top">
          <span class="card-cat-code" style="background:color-mix(in srgb, var(--cat-${cat.id}) 18%, transparent); color:var(--cat-${cat.id})">${cat.code}</span>
          <span class="status-pill"><span class="dot" style="background:var(--status-${project.status})"></span>${status.label}</span>
        </div>
        <h3>${escapeHtml(project.name)}${project.flagship ? ' <span class="flagship-badge">★</span>' : ''}</h3>
        <p class="summary">${escapeHtml(project.summary)}</p>
        <div class="card-tech">${techChips}${moreCount}</div>
      </article>`;
  }

  function renderGrid() {
    const q = state.query.trim();
    const filtered = PROJECTS.filter(
      (p) => (state.category === 'all' || p.category === state.category) && matchesSearch(p, q)
    );

    if (filtered.length === 0) {
      gridEl.innerHTML = `<div class="empty-state"><h3>No projects match</h3><p>Try a different search term or category.</p></div>`;
      return;
    }

    if (state.category !== 'all') {
      gridEl.innerHTML = `<div class="grid">${filtered.map(cardHtml).join('')}</div>`;
    } else {
      // Group by category, preserving CATEGORIES order
      const sections = CATEGORIES.map((cat) => {
        const items = filtered.filter((p) => p.category === cat.id);
        if (!items.length) return '';
        return `
          <div class="category-heading">
            <span class="cat-bar" style="background:var(--cat-${cat.id})"></span>
            <h2>${cat.name}</h2>
            <span class="cat-count">${items.length}</span>
          </div>
          <div class="grid">${items.map(cardHtml).join('')}</div>`;
      }).join('');
      gridEl.innerHTML = sections;
    }

    gridEl.querySelectorAll('.card').forEach((card) => {
      card.addEventListener('click', () => openModal(card.dataset.id));
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openModal(card.dataset.id);
        }
      });
    });
  }

  // ── Modal ─────────────────────────────────────────────────
  function repoLinkHtml(repo) {
    return `<a class="repo-link" href="${repo.url}" target="_blank" rel="noopener noreferrer">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.09.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.22-3.37-1.22-.46-1.19-1.11-1.51-1.11-1.51-.91-.64.07-.62.07-.62 1 .07 1.53 1.05 1.53 1.05.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.06 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.31.1-2.73 0 0 .84-.28 2.75 1.05a9.3 9.3 0 0 1 2.5-.35c.85 0 1.7.12 2.5.35 1.91-1.33 2.75-1.05 2.75-1.05.55 1.42.2 2.47.1 2.73.64.72 1.03 1.63 1.03 2.75 0 3.93-2.34 4.79-4.57 5.05.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.26 10.26 0 0 0 22 12.25C22 6.58 17.52 2 12 2z"/></svg>
      ${escapeHtml(repo.name)}
    </a>`;
  }

  function openModal(id) {
    const project = PROJECTS.find((p) => p.id === id);
    if (!project) return;
    const cat = categoryOf(project.category);
    const status = STATUSES[project.status] || { label: project.status };

    const variantsHtml = project.variants
      ? `<div class="modal-section"><h4>Also in this project</h4><div class="modal-variants">${project.variants
          .map(
            (v) =>
              `<div class="variant-row"><div><div class="v-name">${escapeHtml(v.name)}</div><div class="v-note">${escapeHtml(v.note || '')}</div></div><a href="${v.url}" target="_blank" rel="noopener noreferrer">View →</a></div>`
          )
          .join('')}</div></div>`
      : '';

    const noteHtml = project.note ? `<p class="modal-note">${escapeHtml(project.note)}</p>` : '';

    modalContent.innerHTML = `
      <div class="modal-top">
        <span class="card-cat-code" style="background:color-mix(in srgb, var(--cat-${cat.id}) 18%, transparent); color:var(--cat-${cat.id})">${cat.code}</span>
        <span class="status-pill"><span class="dot" style="background:var(--status-${project.status})"></span>${status.label}</span>
      </div>
      <h2>${escapeHtml(project.name)}</h2>

      <div class="modal-section">
        <h4>What it is</h4>
        <p>${escapeHtml(project.whatItIs)}</p>
      </div>

      <div class="modal-section">
        <h4>Problem it solves</h4>
        <p>${escapeHtml(project.problem)}</p>
        ${noteHtml}
      </div>

      <div class="modal-section">
        <h4>Tech stack</h4>
        <div class="modal-tech">${(project.tech || []).map((t) => `<span class="tech-chip">${escapeHtml(t)}</span>`).join('')}</div>
      </div>

      ${variantsHtml}

      <div class="modal-section">
        <h4>Repository</h4>
        <div class="modal-repos">${(project.repos || []).map(repoLinkHtml).join('')}</div>
      </div>
    `;

    modalBackdrop.classList.add('open');
    document.body.style.overflow = 'hidden';
    history.replaceState(null, '', `#${id}`);
  }

  function closeModal() {
    modalBackdrop.classList.remove('open');
    document.body.style.overflow = '';
    history.replaceState(null, '', location.pathname + location.search);
  }

  modalClose.addEventListener('click', closeModal);
  modalBackdrop.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modalBackdrop.classList.contains('open')) closeModal();
  });

  // ── Search ────────────────────────────────────────────────
  searchEl.addEventListener('input', (e) => {
    state.query = e.target.value;
    renderGrid();
  });

  // ── Init ──────────────────────────────────────────────────
  document.getElementById('year').textContent = new Date().getFullYear();
  renderStats();
  renderPills();
  renderGrid();

  const initialId = location.hash.replace('#', '');
  if (initialId && PROJECTS.some((p) => p.id === initialId)) {
    openModal(initialId);
  }
})();
