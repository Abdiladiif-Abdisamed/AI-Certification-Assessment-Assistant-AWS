const API_URL = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");
const TOKEN_KEY = "ai-cert-assistant-token";

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

async function request(path, options = {}) {
  const { timeout = 30000, auth = true, ...fetchOptions } = options;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(fetchOptions.headers || {}),
  };
  if (auth && token) headers.Authorization = `Bearer ${token}`;

  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });
    const payload = response.status === 204 ? null : await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).join(" ")
        : detail || "The request could not be completed.";
      throw new ApiError(message, response.status);
    }
    return payload;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new ApiError("The request timed out. Check that the backend is running and try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

export const api = {
  register: (body) => request("/auth/register", { method: "POST", body: JSON.stringify(body), auth: false }),
  login: (body) => request("/auth/login", { method: "POST", body: JSON.stringify(body), auth: false }),
  me: () => request("/auth/me"),
  certifications: () => request("/certifications", { auth: false }),
  dashboard: () => request("/dashboard"),
  createAssessment: (body) => request("/assessments", {
    method: "POST",
    body: JSON.stringify(body),
    timeout: 300000,
  }),
  getAssessment: (id) => request(`/assessments/${id}`),
  submitAssessment: (id, answers) => request(`/assessments/${id}/submit`, {
    method: "POST",
    body: JSON.stringify({ answers }),
  }),
  history: () => request("/assessments/history"),
  latestResult: () => request("/results/latest"),
  result: (id) => request(`/results/${id}`),
  weakAreas: () => request("/weak-areas/latest"),
  recommendations: () => request("/recommendations/latest"),
  studyPlan: () => request("/study-plan/latest"),
  readiness: () => request("/readiness/latest"),
  bookmarks: () => request("/bookmarks"),
  addBookmark: (body) => request("/bookmarks", { method: "POST", body: JSON.stringify(body) }),
  deleteBookmark: (id) => request(`/bookmarks/${id}`, { method: "DELETE" }),
  adminOverview: () => request("/admin/overview"),
  adminUsers: () => request("/admin/users"),
  adminExams: () => request("/admin/exams"),
  adminAttempts: (userId) => request(`/admin/attempts${userId ? `?user_id=${userId}` : ""}`),
  adminAssessment: (id) => request(`/admin/assessments/${id}`),
  adminCertification: (id) => request(`/admin/certifications/${id}`),
};
