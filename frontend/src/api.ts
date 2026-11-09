/**
 * Base URL for the Smart ICU backend API.
 * - In local development: defaults to http://127.0.0.1:8000
 * - In production container (served by Nginx reverse proxy): defaults to "" (relative requests)
 * - Can be overridden by setting VITE_API_BASE at build or runtime.
 */
export const API_BASE =
  import.meta.env.VITE_API_BASE !== undefined
    ? import.meta.env.VITE_API_BASE
    : (import.meta.env.PROD ? "" : "http://127.0.0.1:8000");
