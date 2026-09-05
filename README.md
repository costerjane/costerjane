# Coster Jane

A small, modern task board built with **Vite + React + TypeScript**. It supports
adding tasks with a priority, marking them complete, filtering, clearing
completed tasks, and shows live progress. Tasks persist to `localStorage`.

## Tech stack

- [Vite](https://vite.dev/) 8 (dev server + build)
- [React](https://react.dev/) 19 + TypeScript
- [Vitest](https://vitest.dev/) + Testing Library (unit & component tests)
- [oxlint](https://oxc.rs/docs/guide/usage/linter) (linting)

## Getting started

```bash
npm ci          # install dependencies (from package-lock.json)
npm run dev     # start the dev server on http://localhost:5173
```

## Scripts

| Command             | Description                                  |
| ------------------- | -------------------------------------------- |
| `npm run dev`       | Start the Vite dev server (port 5173).       |
| `npm run build`     | Type-check then build for production.        |
| `npm run preview`   | Preview the production build locally.         |
| `npm run lint`      | Lint the codebase with oxlint.               |
| `npm run typecheck` | Type-check with `tsc -b` (no emit).          |
| `npm run test`      | Run the unit & component tests once.         |
| `npm run test:watch`| Run the tests in watch mode.                 |

## Project structure

```
src/
  App.tsx          # Task board UI
  App.css          # Component styles
  lib/tasks.ts     # Pure task logic (add/toggle/remove/filter/summarize)
  lib/tasks.test.ts# Unit tests for the task logic
  App.test.tsx     # Component tests for the UI
  test/setup.ts    # Test environment setup
```

## Cloud Agent environment

This repository is configured for Cursor Cloud Agents via
[`.cursor/environment.json`](.cursor/environment.json):

- **install:** `npm ci`
- **terminals:** runs `npm run dev` so the app is available on port `5173`.
