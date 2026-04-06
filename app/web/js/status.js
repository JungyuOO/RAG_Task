// app/web/js/status.js
class StatusBar {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.isExpanded = false;
    this.stages = [];
    this._render();
  }

  _render() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="status-bar">
        <div class="status-header" onclick="statusBar && statusBar.toggle()">
          <span class="status-current">${this.stages.length > 0 ? this.stages[this.stages.length - 1].message : '처리중...'}</span>
          <span class="status-toggle">${this.isExpanded ? '▼' : '▲'}</span>
        </div>
        <div class="status-details" style="display:${this.isExpanded ? 'block' : 'none'}">
          ${this.stages.map(s => `
            <div class="status-stage ${s.active ? 'active' : ''} ${s.done ? 'done' : ''}">
              <span class="status-icon">${s.done ? '✓' : s.active ? '◌' : '○'}</span>
              ${s.message}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  addStage(stage, message) {
    this.stages.forEach(s => { s.active = false; s.done = true; });
    this.stages.push({ stage, message, active: true, done: false });
    this._render();
  }

  complete() {
    this.stages.forEach(s => { s.active = false; s.done = true; });
    this._render();
    setTimeout(() => { if (this.container) this.container.style.display = 'none'; }, 1500);
  }

  toggle() {
    this.isExpanded = !this.isExpanded;
    this._render();
  }

  show() {
    this.stages = [];
    if (this.container) this.container.style.display = 'block';
    this._render();
  }
}

let statusBar = null;
function initStatusBar() {
  statusBar = new StatusBar('status-bar-container');
}
