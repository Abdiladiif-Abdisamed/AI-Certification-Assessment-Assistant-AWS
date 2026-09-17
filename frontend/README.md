# React frontend

React 19 + Vite learner dashboard connected to the FastAPI API. It includes
local registration/login, dashboard metrics, certification readiness, grounded
assessment generation, assessment submission, result/source review, weak areas,
recommendations, study plan, history, bookmarks, themes, and responsive mobile
navigation.

Start the API first, then:

```powershell
cd frontend
npm install
npm run dev
```

The default API path is `/api`. During local development, Vite proxies that
path to `http://127.0.0.1:8000`, so the frontend and API can also be shared
through one temporary public frontend URL. Set `VITE_API_URL` only when the
frontend must call a separately hosted API origin.

Verify the frontend:

```powershell
npm run lint
npm run build
```

Full browser end-to-end automation is intentionally deferred.
