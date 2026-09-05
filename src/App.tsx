import { useEffect, useMemo, useState } from 'react'
import './App.css'
import {
  addTask,
  clearDone,
  filterTasks,
  removeTask,
  summarize,
  toggleTask,
  type Task,
  type TaskFilter,
  type TaskPriority,
} from './lib/tasks'

const STORAGE_KEY = 'costerjane.tasks.v1'

const SEED_TASKS: Task[] = [
  {
    id: 'seed-1',
    title: 'Welcome to Coster Jane — add your first task',
    priority: 'high',
    done: false,
    createdAt: Date.now(),
  },
  {
    id: 'seed-2',
    title: 'Click a task to mark it complete',
    priority: 'medium',
    done: false,
    createdAt: Date.now() - 1,
  },
  {
    id: 'seed-3',
    title: 'Read the project setup guide',
    priority: 'low',
    done: true,
    createdAt: Date.now() - 2,
  },
]

function loadTasks(): Task[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return SEED_TASKS
    const parsed = JSON.parse(raw) as Task[]
    return Array.isArray(parsed) ? parsed : SEED_TASKS
  } catch {
    return SEED_TASKS
  }
}

const FILTERS: TaskFilter[] = ['all', 'active', 'done']
const PRIORITIES: TaskPriority[] = ['low', 'medium', 'high']

function App() {
  const [tasks, setTasks] = useState<Task[]>(loadTasks)
  const [title, setTitle] = useState('')
  const [priority, setPriority] = useState<TaskPriority>('medium')
  const [filter, setFilter] = useState<TaskFilter>('all')

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks))
    } catch {
      // Ignore storage failures (e.g. private mode).
    }
  }, [tasks])

  const visible = useMemo(
    () => filterTasks(tasks, filter),
    [tasks, filter],
  )
  const stats = useMemo(() => summarize(tasks), [tasks])

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    if (title.trim().length === 0) return
    setTasks((current) => addTask(current, title, priority))
    setTitle('')
    setPriority('medium')
  }

  return (
    <main className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            CJ
          </span>
          <div>
            <h1>Coster Jane</h1>
            <p className="tagline">A focused task board for getting things done.</p>
          </div>
        </div>
        <dl className="stats" aria-label="Task statistics">
          <div className="stat">
            <dt>Active</dt>
            <dd>{stats.active}</dd>
          </div>
          <div className="stat">
            <dt>Done</dt>
            <dd>{stats.done}</dd>
          </div>
          <div className="stat">
            <dt>Progress</dt>
            <dd>{stats.completionRate}%</dd>
          </div>
        </dl>
      </header>

      <div className="progress-track" aria-hidden="true">
        <div
          className="progress-fill"
          style={{ width: `${stats.completionRate}%` }}
        />
      </div>

      <form className="add-form" onSubmit={handleSubmit}>
        <input
          className="add-input"
          type="text"
          value={title}
          placeholder="What needs doing?"
          aria-label="Task title"
          onChange={(event) => setTitle(event.target.value)}
        />
        <select
          className="add-priority"
          value={priority}
          aria-label="Priority"
          onChange={(event) =>
            setPriority(event.target.value as TaskPriority)
          }
        >
          {PRIORITIES.map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </select>
        <button className="add-button" type="submit">
          Add task
        </button>
      </form>

      <div className="toolbar">
        <div className="filters" role="tablist" aria-label="Filter tasks">
          {FILTERS.map((option) => (
            <button
              key={option}
              type="button"
              role="tab"
              aria-selected={filter === option}
              className={filter === option ? 'filter active' : 'filter'}
              onClick={() => setFilter(option)}
            >
              {option}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="clear-button"
          disabled={stats.done === 0}
          onClick={() => setTasks((current) => clearDone(current))}
        >
          Clear completed
        </button>
      </div>

      <ul className="task-list">
        {visible.length === 0 && (
          <li className="empty">Nothing here yet. Add a task above.</li>
        )}
        {visible.map((task) => (
          <li
            key={task.id}
            className={task.done ? 'task done' : 'task'}
            data-testid="task-item"
          >
            <label className="task-main">
              <input
                type="checkbox"
                checked={task.done}
                onChange={() =>
                  setTasks((current) => toggleTask(current, task.id))
                }
                aria-label={`Mark "${task.title}" as ${task.done ? 'active' : 'done'}`}
              />
              <span className="task-title">{task.title}</span>
            </label>
            <span className={`badge badge-${task.priority}`}>
              {task.priority}
            </span>
            <button
              type="button"
              className="delete-button"
              aria-label={`Delete "${task.title}"`}
              onClick={() => setTasks((current) => removeTask(current, task.id))}
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      <footer className="app-footer">
        {stats.total} task{stats.total === 1 ? '' : 's'} · {stats.active} active
      </footer>
    </main>
  )
}

export default App
