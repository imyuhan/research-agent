const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function baseUrl() {
  return DEFAULT_BASE_URL.replace(/\/$/, "");
}

async function request(path, options = {}) {
  const response = await fetch(`${baseUrl()}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

export function checkHealth() {
  return request("/health");
}

export function createResearch(topic) {
  return request("/api/research", {
    method: "POST",
    body: JSON.stringify({ topic })
  });
}

export function getResearch(jobId) {
  return request(`/api/research/${jobId}`);
}

export function getResearchReport(jobId) {
  return request(`/api/research/${jobId}/report`);
}

export function getResearchEvents(jobId) {
  return request(`/api/research/${jobId}/events/history`);
}

export function cancelResearch(jobId) {
  return request(`/api/research/${jobId}/cancel`, {
    method: "POST"
  });
}

export function subscribeResearchEvents(jobId, after = 0, handlers = {}) {
  const url = `${baseUrl()}/api/research/${jobId}/events?after=${after}`;
  const source = new EventSource(url);

  source.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handlers.onEvent?.(payload);
      if (["job.completed", "job.failed", "job.cancelled"].includes(payload.type)) {
        handlers.onDone?.(payload);
      }
    } catch (error) {
      handlers.onError?.(error);
    }
  };

  source.onerror = (error) => {
    handlers.onError?.(error);
  };

  return source;
}
