import { describe, expect, it } from 'vitest'
import {
  addTask,
  clearDone,
  createTask,
  filterTasks,
  removeTask,
  summarize,
  toggleTask,
  type Task,
} from './tasks'

function seed(): Task[] {
  return [
    { id: 'a', title: 'A', priority: 'high', done: false, createdAt: 1 },
    { id: 'b', title: 'B', priority: 'low', done: true, createdAt: 2 },
  ]
}

describe('addTask', () => {
  it('prepends a trimmed task', () => {
    const result = addTask(seed(), '  Buy milk  ', 'high')
    expect(result).toHaveLength(3)
    expect(result[0].title).toBe('Buy milk')
    expect(result[0].priority).toBe('high')
    expect(result[0].done).toBe(false)
  })

  it('ignores blank titles', () => {
    const tasks = seed()
    expect(addTask(tasks, '   ')).toBe(tasks)
  })
})

describe('toggleTask', () => {
  it('flips the done flag for the matching id only', () => {
    const result = toggleTask(seed(), 'a')
    expect(result[0].done).toBe(true)
    expect(result[1].done).toBe(true)
  })
})

describe('removeTask', () => {
  it('removes the matching task', () => {
    const result = removeTask(seed(), 'a')
    expect(result.map((t) => t.id)).toEqual(['b'])
  })
})

describe('clearDone', () => {
  it('drops completed tasks', () => {
    expect(clearDone(seed()).map((t) => t.id)).toEqual(['a'])
  })
})

describe('filterTasks', () => {
  it('filters by status', () => {
    const tasks = seed()
    expect(filterTasks(tasks, 'all')).toHaveLength(2)
    expect(filterTasks(tasks, 'active').map((t) => t.id)).toEqual(['a'])
    expect(filterTasks(tasks, 'done').map((t) => t.id)).toEqual(['b'])
  })
})

describe('summarize', () => {
  it('computes counts and completion rate', () => {
    expect(summarize(seed())).toEqual({
      total: 2,
      done: 1,
      active: 1,
      completionRate: 50,
    })
  })

  it('handles an empty list', () => {
    expect(summarize([])).toEqual({
      total: 0,
      done: 0,
      active: 0,
      completionRate: 0,
    })
  })
})

describe('createTask', () => {
  it('generates unique ids', () => {
    expect(createTask('one').id).not.toBe(createTask('two').id)
  })
})
