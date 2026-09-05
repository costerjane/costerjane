export type TaskPriority = 'low' | 'medium' | 'high'

export interface Task {
  id: string
  title: string
  priority: TaskPriority
  done: boolean
  createdAt: number
}

export type TaskFilter = 'all' | 'active' | 'done'

let idSeed = 0

/** Create a task with a stable-ish unique id. */
export function createTask(
  title: string,
  priority: TaskPriority = 'medium',
): Task {
  idSeed += 1
  return {
    id: `${Date.now().toString(36)}-${idSeed}`,
    title: title.trim(),
    priority,
    done: false,
    createdAt: Date.now(),
  }
}

/** Append a new task, ignoring empty titles. Returns a new array. */
export function addTask(
  tasks: Task[],
  title: string,
  priority: TaskPriority = 'medium',
): Task[] {
  if (title.trim().length === 0) return tasks
  return [createTask(title, priority), ...tasks]
}

export function toggleTask(tasks: Task[], id: string): Task[] {
  return tasks.map((task) =>
    task.id === id ? { ...task, done: !task.done } : task,
  )
}

export function removeTask(tasks: Task[], id: string): Task[] {
  return tasks.filter((task) => task.id !== id)
}

export function clearDone(tasks: Task[]): Task[] {
  return tasks.filter((task) => !task.done)
}

export function filterTasks(tasks: Task[], filter: TaskFilter): Task[] {
  switch (filter) {
    case 'active':
      return tasks.filter((task) => !task.done)
    case 'done':
      return tasks.filter((task) => task.done)
    default:
      return tasks
  }
}

export interface TaskStats {
  total: number
  done: number
  active: number
  completionRate: number
}

export function summarize(tasks: Task[]): TaskStats {
  const total = tasks.length
  const done = tasks.filter((task) => task.done).length
  const active = total - done
  const completionRate = total === 0 ? 0 : Math.round((done / total) * 100)
  return { total, done, active, completionRate }
}
