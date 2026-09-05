import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import App from './App'

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  localStorage.clear()
})

describe('<App />', () => {
  it('adds a task through the form', async () => {
    const user = userEvent.setup()
    render(<App />)

    const before = screen.getAllByTestId('task-item').length
    await user.type(screen.getByLabelText('Task title'), 'Ship the release')
    await user.click(screen.getByRole('button', { name: 'Add task' }))

    expect(screen.getAllByTestId('task-item')).toHaveLength(before + 1)
    expect(screen.getByText('Ship the release')).toBeInTheDocument()
  })

  it('completes a task and updates progress', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText('Task title'), 'Write docs')
    await user.click(screen.getByRole('button', { name: 'Add task' }))
    await user.click(
      screen.getByLabelText('Mark "Write docs" as done'),
    )

    expect(
      screen.getByLabelText('Mark "Write docs" as active'),
    ).toBeChecked()
  })
})
