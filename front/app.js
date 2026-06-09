const videoInput = document.querySelector("#videoInput");
const dropZone = document.querySelector("#dropZone");
const videoPreview = document.querySelector("#videoPreview");
const videoShell = document.querySelector("#videoShell");
const analyzeButton = document.querySelector("#analyzeButton");
const resetButton = document.querySelector("#resetButton");
const statusText = document.querySelector("#statusText");
const statusBadge = document.querySelector("#statusBadge");
const loader = document.querySelector("#loader");
const summaryCards = document.querySelector("#summaryCards");
const jumpList = document.querySelector("#jumpList");
const totalJumps = document.querySelector("#totalJumps");
const problemJumps = document.querySelector("#problemJumps");

let selectedVideoUrl = "";
let selectedVideoFile = null;
let lastResult = null;

const USE_MOCK = false;
// пусто = тот же origin, что и страница (backend сам раздаёт фронт)
const API_BASE = "";

// анализ полного видео асинхронный: POST /analyze -> job_id, затем поллинг /jobs/{id}
const JOB_POLL_INTERVAL_MS = 4000;

const mockResults = [
  {
    jumps: [
      {
        jump_type: "Аксель",
        rotation_status: "докрут",
        fall: false,
        rotations: 2.5,
        start_time: "00:12.40",
        end_time: "00:13.90",
      },
      {
        jump_type: "Тулуп",
        rotation_status: "недокрут",
        fall: false,
        rotations: 3.25,
        start_time: "00:31.10",
        end_time: "00:32.80",
      },
      {
        jump_type: "Сальхов",
        rotation_status: "докрут",
        fall: false,
        rotations: 3,
        start_time: "00:47.60",
        end_time: "00:49.20",
      },
    ],
  },
  {
    jumps: [
      {
        jump_type: "Лутц",
        rotation_status: "докрут",
        fall: false,
        rotations: 3,
        start_time: "00:08.20",
        end_time: "00:09.70",
      },
      {
        jump_type: "Флип",
        rotation_status: "недокрут",
        fall: true,
        rotations: 3.25,
        start_time: "00:26.50",
        end_time: "00:28.10",
      },
      {
        jump_type: "Риттбергер",
        rotation_status: "докрут",
        fall: false,
        rotations: 2,
        start_time: "00:52.00",
        end_time: "00:53.30",
      },
      {
        jump_type: "Аксель",
        rotation_status: "докрут",
        fall: false,
        rotations: 2.5,
        start_time: "01:14.40",
        end_time: "01:16.00",
      },
    ],
  },
  {
    jumps: [
      {
        jump_type: "Тулуп",
        rotation_status: "докрут",
        fall: false,
        rotations: 4,
        start_time: "00:15.00",
        end_time: "00:16.80",
      },
      {
        jump_type: "Сальхов",
        rotation_status: "докрут",
        fall: false,
        rotations: 3,
        start_time: "00:39.30",
        end_time: "00:40.90",
      },
      {
        jump_type: "Аксель",
        rotation_status: "недокрут",
        fall: false,
        rotations: 2.25,
        start_time: "01:02.10",
        end_time: "01:03.70",
      },
      {
        jump_type: "Флип",
        rotation_status: "докрут",
        fall: false,
        rotations: 3,
        start_time: "01:28.60",
        end_time: "01:30.00",
      },
      {
        jump_type: "Лутц",
        rotation_status: "докрут",
        fall: true,
        rotations: 3,
        start_time: "01:51.20",
        end_time: "01:53.10",
      },
    ],
  },
];

function setStatus(text, badge, state = "idle") {
  statusText.textContent = text;
  statusBadge.textContent = badge;
  statusBadge.classList.toggle("is-loading", state === "loading");
  statusBadge.classList.toggle("is-ready", state === "ready");
}

function handleVideo(file) {
  if (!file || !file.type.startsWith("video/")) {
    alert("Пожалуйста, выберите видеофайл.");
    return;
  }

  if (selectedVideoUrl) {
    URL.revokeObjectURL(selectedVideoUrl);
  }

  selectedVideoUrl = URL.createObjectURL(file);
  selectedVideoFile = file;
  videoPreview.src = selectedVideoUrl;
  dropZone.hidden = true;
  videoShell.classList.add("is-visible");
  analyzeButton.disabled = false;
  resetButton.disabled = false;
  summaryCards.hidden = true;
  jumpList.hidden = true;
  jumpList.innerHTML = "";
  loader.hidden = true;
  lastResult = null;

  setStatus(`Выбрано видео: ${file.name}`, "Готово");
}

function getJumpStart(jump) {
  return jump.start_time ?? jump.start_timecode ?? jump.start ?? "-";
}

function getJumpEnd(jump) {
  return jump.end_time ?? jump.end_timecode ?? jump.end ?? "-";
}

function parseTimecode(value) {
  if (typeof value === "number") {
    return value;
  }

  if (!value || typeof value !== "string") {
    return null;
  }

  const parts = value.split(":").map(Number);

  if (parts.some(Number.isNaN)) {
    return null;
  }

  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  }

  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }

  return null;
}

function escapeHtml(value) {
  return String(value ?? "-")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getNormalizedJumps(result) {
  const jumps = Array.isArray(result?.jumps) ? result.jumps : [result];
  return jumps.filter(Boolean).slice(0, 50);
}

function createJumpCard(jump, index) {
  const item = document.createElement("article");
  item.className = "jump-card";

  const start = getJumpStart(jump);
  const end = getJumpEnd(jump);
  const hasProblem = jump.fall || jump.rotation_status === "недокрут";
  item.innerHTML = `
    <div class="jump-card-header">
      <span class="jump-number">#${index + 1}</span>
      <strong>${escapeHtml(jump.jump_type ?? "Прыжок")}</strong>
      <span class="jump-badge ${hasProblem ? "is-problem" : "is-clean"}">
        ${hasProblem ? "есть ошибка" : "чисто"}
      </span>
    </div>
    <div class="jump-grid">
      <div>
        <span>Таймкод</span>
        <strong>${escapeHtml(start)} - ${escapeHtml(end)}</strong>
      </div>
      <div>
        <span>Обороты</span>
        <strong>${escapeHtml(jump.rotations ?? "-")}</strong>
      </div>
      <div>
        <span>Докрут</span>
        <strong class="${jump.rotation_status === "докрут" ? "is-good" : "is-bad"}">
          ${escapeHtml(jump.rotation_status ?? "-")}
        </strong>
      </div>
      <div>
        <span>Падение</span>
        <strong class="${jump.fall ? "is-bad" : "is-good"}">
          ${jump.fall ? "упал" : "не упал"}
        </strong>
      </div>
    </div>
  `;

  return item;
}

function renderResult(result) {
  const jumps = getNormalizedJumps(result);
  const problemCount = jumps.filter((jump) => jump.fall || jump.rotation_status === "недокрут").length;

  totalJumps.textContent = jumps.length;
  problemJumps.textContent = problemCount;

  jumpList.innerHTML = "";
  jumps.forEach((jump, index) => {
    jumpList.append(createJumpCard(jump, index));
  });

  summaryCards.hidden = false;
  jumpList.hidden = jumps.length === 0;
}

function getMockResult() {
  const result = mockResults[Math.floor(Math.random() * mockResults.length)];

  return {
    ...result,
    total_jumps: result.jumps.length,
    analyzed_at: new Date().toISOString(),
    model_version: "frontend-mock-v1",
  };
}

async function analyzeVideo() {
  if (!selectedVideoFile) {
    return;
  }

  analyzeButton.disabled = true;
  loader.hidden = false;
  summaryCards.hidden = true;
  jumpList.hidden = true;
  setStatus("Загрузка видео...", "Загрузка", "loading");

  if (USE_MOCK) {
    window.setTimeout(() => {
      lastResult = getMockResult();
      loader.hidden = true;
      analyzeButton.disabled = false;
      setStatus(`Найдено прыжков: ${lastResult.jumps.length}`, "Готово", "ready");
      renderResult(lastResult);
    }, 1400);
    return;
  }

  try {
    const { job_id: jobId } = await uploadVideo(selectedVideoFile);
    setStatus("Видео анализируется...", "Анализ", "loading");
    lastResult = await pollJob(jobId);

    loader.hidden = true;
    analyzeButton.disabled = false;
    setStatus(`Найдено прыжков: ${getNormalizedJumps(lastResult).length}`, "Готово", "ready");
    renderResult(lastResult);
  } catch (error) {
    loader.hidden = true;
    analyzeButton.disabled = false;
    setStatus(error.message || "Ошибка анализа видео.", "Ошибка");
  }
}

// Загрузка видео через XMLHttpRequest — в отличие от fetch, отдаёт прогресс
// аплоада (upload.onprogress). Статус-строка показывает «X/Y МБ (Z%)».
function uploadVideo(file) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("video", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/analyze`);

    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }
      const mb = 1024 * 1024;
      const pct = Math.round((event.loaded / event.total) * 100);
      setStatus(
        `Загрузка видео: ${(event.loaded / mb).toFixed(0)}/${(event.total / mb).toFixed(0)} МБ (${pct}%)`,
        "Загрузка",
        "loading",
      );
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (error) {
          reject(new Error("Некорректный ответ сервера."));
        }
      } else {
        reject(new Error("Не удалось поставить видео в обработку."));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Ошибка сети при загрузке видео.")));
    xhr.addEventListener("abort", () => reject(new Error("Загрузка прервана.")));

    xhr.send(formData);
  });
}

// Поллит /jobs/{id} до завершения. Анализ полного видео идёт минуты,
// поэтому ждём столько, сколько нужно.
async function pollJob(jobId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL_MS));

    const response = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!response.ok) {
      throw new Error("Не удалось получить статус задачи.");
    }

    const job = await response.json();
    if (job.status === "done") {
      return job.result;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Анализ завершился ошибкой.");
    }
    // queued / processing — продолжаем ждать
  }
}

function resetApp() {
  if (selectedVideoUrl) {
    URL.revokeObjectURL(selectedVideoUrl);
  }

  selectedVideoUrl = "";
  selectedVideoFile = null;
  lastResult = null;
  videoInput.value = "";
  videoPreview.removeAttribute("src");
  videoPreview.load();
  dropZone.hidden = false;
  videoShell.classList.remove("is-visible");
  analyzeButton.disabled = true;
  resetButton.disabled = true;
  loader.hidden = true;
  summaryCards.hidden = true;
  jumpList.hidden = true;
  jumpList.innerHTML = "";
  setStatus("Загрузите программу, чтобы начать анализ.", "Ожидание");
}

videoInput.addEventListener("change", (event) => {
  handleVideo(event.target.files[0]);
});

analyzeButton.addEventListener("click", analyzeVideo);
resetButton.addEventListener("click", resetApp);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  handleVideo(event.dataTransfer.files[0]);
});
