import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Plus, Edit2, Plus as PlusIcon, Save, X } from 'lucide-react'
import { SubTask, Goal, TaskMetadata, Tag, Habit } from './types'

function App() {
  const [goals, setGoals] = useState<Goal[]>([])
  const [editingGoal, setEditingGoal] = useState<string | null>(null)
  const [editingSubTask, setEditingSubTask] = useState<{goalId: string, taskId: string} | null>(null)

  const addNewGoal = () => {
    const newGoal: Goal = {
      id: Date.now().toString(),
      title: '新しい目標',
      currentValue: 0,
      subTasks: []
    }
    setGoals([...goals, newGoal])
    setEditingGoal(newGoal.id) // Start editing the title immediately
  }

  const updateGoalValue = (goalId: string, value: number) => {
    setGoals(goals.map(goal => 
      goal.id === goalId ? { ...goal, currentValue: value } : goal
    ))
  }

  const updateGoalTitle = (goalId: string, title: string) => {
    setGoals(goals.map(goal =>
      goal.id === goalId ? { ...goal, title } : goal
    ))
    setEditingGoal(null)
  }

  const addSubTask = (goalId: string) => {
    const newTaskId = Date.now().toString()
    setGoals(goals.map(goal =>
      goal.id === goalId
        ? {
            ...goal,
            subTasks: [
              ...goal.subTasks,
              { id: newTaskId, title: '新しいサブタスク', completed: false }
            ]
          }
        : goal
    ))
    setEditingSubTask({ goalId, taskId: newTaskId })
  }

  const updateSubTaskTitle = (goalId: string, taskId: string, title: string) => {
    setGoals(goals.map(goal =>
      goal.id === goalId
        ? {
            ...goal,
            subTasks: goal.subTasks.map(task =>
              task.id === taskId ? { ...task, title } : task
            )
          }
        : goal
    ))
    setEditingSubTask(null)
  }

  const toggleSubTask = (goalId: string, taskId: string) => {
    setGoals(goals.map(goal =>
      goal.id === goalId
        ? {
            ...goal,
            subTasks: goal.subTasks.map(task =>
              task.id === taskId ? { ...task, completed: !task.completed } : task
            )
          }
        : goal
    ))
  }

  const deleteSubTask = (goalId: string, taskId: string) => {
    setGoals(goals.map(goal =>
      goal.id === goalId
        ? {
            ...goal,
            subTasks: goal.subTasks.filter(task => task.id !== taskId)
          }
        : goal
    ))
  }

  return (
    <div className="container mx-auto p-4">
      <header className="sticky top-0 z-10 bg-white flex justify-between items-center mb-6 py-4 border-b">
        <h1 className="text-2xl font-bold">スキル習熟度トラッカー</h1>
        <Button onClick={addNewGoal} className="flex items-center gap-2">
          <Plus className="w-4 h-4" />
          新しい目標を追加
        </Button>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {goals.map(goal => (
          <Card key={goal.id} className="shadow-sm hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                {editingGoal === goal.id ? (
                  <div className="flex w-full gap-2">
                    <Input
                      value={goal.title}
                      onChange={(e) => updateGoalTitle(goal.id, e.target.value)}
                      autoFocus
                      className="flex-1"
                      placeholder="目標名を入力"
                    />
                    <Button
                      size="sm"
                      onClick={() => setEditingGoal(null)}
                      variant="outline"
                    >
                      <Save className="w-4 h-4" />
                    </Button>
                  </div>
                ) : (
                  <CardTitle className="flex items-center justify-between w-full">
                    <span className="truncate">{goal.title}</span>
                    <button
                      onClick={() => setEditingGoal(goal.id)}
                      className="p-1 hover:bg-gray-100 rounded"
                    >
                      <Edit2 className="w-4 h-4" />
                    </button>
                  </CardTitle>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">現在の値</label>
                  <Input
                    type="number"
                    value={goal.currentValue}
                    onChange={(e) => updateGoalValue(goal.id, Number(e.target.value))}
                    className="mt-1"
                    min="0"
                  />
                </div>
                <div>
                  <h3 className="text-sm font-medium mb-2">サブタスク</h3>
                  <div className="space-y-2">
                    {goal.subTasks.map(task => (
                      <div key={task.id} className="flex items-center gap-2 group">
                        <input
                          type="checkbox"
                          checked={task.completed}
                          onChange={() => toggleSubTask(goal.id, task.id)}
                          className="rounded"
                        />
                        {editingSubTask?.goalId === goal.id && editingSubTask?.taskId === task.id ? (
                          <div className="flex flex-1 gap-2">
                            <Input
                              value={task.title}
                              onChange={(e) => updateSubTaskTitle(goal.id, task.id, e.target.value)}
                              autoFocus
                              className="flex-1"
                              placeholder="サブタスク名を入力"
                            />
                            <Button
                              size="sm"
                              onClick={() => setEditingSubTask(null)}
                              variant="outline"
                            >
                              <Save className="w-4 h-4" />
                            </Button>
                          </div>
                        ) : (
                          <div className="flex flex-1 items-center justify-between">
                            <span
                              className={`flex-1 ${task.completed ? 'line-through text-gray-500' : ''}`}
                              onClick={() => setEditingSubTask({ goalId: goal.id, taskId: task.id })}
                            >
                              {task.title}
                            </span>
                            <button
                              onClick={() => deleteSubTask(goal.id, task.id)}
                              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-100 rounded"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => addSubTask(goal.id)}
                      className="w-full mt-2"
                    >
                      <PlusIcon className="w-4 h-4 mr-2" />
                      サブタスクを追加
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

export default App
