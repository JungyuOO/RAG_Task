class StatusBar {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.isExpanded = false;
    this.stages = [];
    this.visible = false;
    this._hideTimer = null;
    this._render();
  }

  _render() {
    if (!this.container) return;
    const currentMessage = this.stages.length > 0 ? this.stages[this.stages.length - 1].message : "처리중...";
    this.container.innerHTML = `
      <div class="status-bar-shell">
        <button class="status-header" type="button">
          <span class="status-current">${currentMessage}</span>
          <span class="status-toggle">${this.isExpanded ? "접기" : "펼치기"}</span>
        </button>
        <div class="status-details" style="display:${this.isExpanded ? "block" : "none"}">
          ${this.stages.map((stage) => `
            <div class="status-stage ${stage.active ? "active" : ""} ${stage.done ? "done" : ""}">
              <span class="status-icon">${stage.done ? "완료" : stage.active ? "진행" : "대기"}</span>
              <span class="status-text">${stage.message}</span>
            </div>
          `).join("")}
        </div>
      </div>
    `;
    this.container.style.display = this.visible ? "block" : "none";
    const header = this.container.querySelector(".status-header");
    if (header) {
      header.addEventListener("click", () => this.toggle());
    }
  }

  reset() {
    if (this._hideTimer) {
      clearTimeout(this._hideTimer);
      this._hideTimer = null;
    }
    this.stages = [];
    this.visible = true;
    this.isExpanded = false;
    this._render();
  }

  show() {
    if (!this.visible) {
      this.visible = true;
      this._render();
    }
  }

  addStage(stage, message) {
    const existing = this.stages.find((item) => item.stage === stage);
    this.stages.forEach((item) => {
      item.active = false;
      item.done = true;
    });
    if (existing) {
      existing.message = message;
      existing.active = true;
      existing.done = false;
    } else {
      this.stages.push({ stage, message, active: true, done: false });
    }
    this.visible = true;
    this._render();
  }

  complete() {
    this.stages.forEach((item) => {
      item.active = false;
      item.done = true;
    });
    this.visible = true;
    this._render();
    if (this._hideTimer) clearTimeout(this._hideTimer);
    this._hideTimer = setTimeout(() => {
      this.visible = false;
      this._render();
    }, 1500);
  }

  toggle() {
    this.isExpanded = !this.isExpanded;
    this._render();
  }
}

let statusBar = null;
function initStatusBar() {
  statusBar = new StatusBar("status-bar-container");
}
