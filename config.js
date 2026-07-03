// Configuration for the frontend.
//
// Local development: leave as-is (points at your local backend on :4000).
//
// Once your backend is deployed to Render, set backendUrl below to your
// Render URL + /api/upload, e.g.:
//   https://your-app-name.onrender.com/api/upload
window.EXPRESSION_APP_CONFIG = {
  backendUrl: "http://localhost:4000/api/upload",
};
