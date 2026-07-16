/* =========================================================
   FLTalk - Global App Configuration
   ========================================================= */

const CONFIG = {
  // 🔗 CHANGE THESE VALUES ONLY
  GITHUB_URL: "https://github.com/ali-liyakat/fltalk_system",
  // SERVER_URL: "https://fltalk-system.onrender.com",
  SERVER_URL: "http://127.0.0.1:8000",
  RUN_EXPERIMENT_PAGE: "./experiment.html"
};

/* =========================================================
   Utility helpers
   ========================================================= */

function $(id) {
  return document.getElementById(id);
}

/* =========================================================
   Download + Run Experiment buttons
   ========================================================= */

const downloadBtn = $("downloadBtn");
if (downloadBtn) {
  downloadBtn.href = CONFIG.GITHUB_URL;
}

const runBtn = $("runBtn");
if (runBtn) {
  runBtn.href = CONFIG.RUN_EXPERIMENT_PAGE;
}

/* =========================================================
   Server URL section
   ========================================================= */

const serverUrlEl = $("serverUrl");
if (serverUrlEl) {
  serverUrlEl.textContent = CONFIG.SERVER_URL;
}

const copyBtn = $("copyBtn");
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const originalText = copyBtn.textContent;

    try {
      await navigator.clipboard.writeText(CONFIG.SERVER_URL);
      copyBtn.textContent = "Copied ✓";
      setTimeout(() => {
        copyBtn.textContent = originalText;
      }, 900);
    } catch (err) {
      alert("Copy failed. Please copy manually:\n" + CONFIG.SERVER_URL);
    }
  });
}

/* =========================================================
   Footer: Year + GitHub link
   ========================================================= */

const yearEl = $("year");
if (yearEl) {
  yearEl.textContent = new Date().getFullYear();
}

const githubLink = $("githubLink");
if (githubLink) {
  githubLink.href = CONFIG.GITHUB_URL;
}

/* =========================================================
   (Optional) Future hooks – safe placeholders
   ========================================================= */

// Example: later you can add log streaming here
// Example: server health check
// Example: experiment status polling

console.log("FLTalk frontend initialized successfully.");
