import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Edit2, Save, Plus as PlusIcon, X } from 'lucide-react'
import { Goal, SubTask } from '../types'

interface GoalCardProps {
  goal: Goal
  onUpdate: (goalId: string, updates: Partial<Goal>) => void
  onUpdateSubTask: (goalId: string, taskId: string, updates: Partial<SubTask>) => void
  onAddSubTask: (goalId: string) => void
  onDeleteSubTask: (goalId: string, taskId: string) => void
}

export const GoalCard: React.FC<GoalCardProps> = ({
  goal,
  onUpdate,
  onUpdateSubTask,
  onAddSubTask,
  onDeleteSubTask,
}) => {
  const [isEditing, setIsEditing] = React.useState(false)
  const [editingTaskId, setEditingTaskId] = React.useState<string | null>(null)

  const handleUpdateTitle = (title: string) => {
    onUpdate(goal.id, { title })
    setIsEditing(false)
  }

  const handleUpdateValue = (value: number) => {
    onUpdate(goal.id, { currentValue: value })
  }

  const handleUpdateSubTask = (taskId: string, title: string) => {
    onUpdateSubTask(goal.id, taskId, { title })
    setEditingTaskId(null)
  }

  return (
    <Card className="shadow-sm hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          {isEditing ? (
            <div className="flex w-full gap-2">
              <Input
                value={goal.title}
                onChange={(e) => handleUpdateTitle(e.target.value)}
                autoFocus
                className="flex-1"
                placeholder="目標名を入力"
              />
              <Button
                size="sm"
                onClick={() => setIsEditing(false)}
                variant="outline"
              >
                <Save className="w-4 h-4" />
              </Button>
            </div>
          ) : (
            <CardTitle className="flex items-center justify-between w-full">
              <span className="truncate">{goal.title}</span>
              <button
                onClick={() => setIsEditing(true)}
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
              onChange={(e) => handleUpdateValue(Number(e.target.value))}
              className="mt-1"
              min="0"
            />
          </div>
          <div>
            <h3 className="text-sm font-medium mb-2">サブタスク</h3>
            <div className="space-y-2">
              {goal.subTasks.map((task) => (
                <div key={task.id} className="flex items-center gap-2 group">
                  <input
                    type="checkbox"
                    checked={task.completed}
                    onChange={() => onUpdateSubTask(goal.id, task.id, { completed: !task.completed })}
                    className="rounded"
                  />
                  {editingTaskId === task.id ? (
                    <div className="flex flex-1 gap-2">
                      <Input
                        value={task.title}
                        onChange={(e) => handleUpdateSubTask(task.id, e.target.value)}
                        autoFocus
                        className="flex-1"
                        placeholder="サブタスク名を入力"
                      />
                      <Button
                        size="sm"
                        onClick={() => setEditingTaskId(null)}
                        variant="outline"
                      >
                        <Save className="w-4 h-4" />
                      </Button>
                    </div>
                  ) : (
                    <div className="flex flex-1 items-center justify-between">
                      <span
                        className={`flex-1 ${task.completed ? 'line-through text-gray-500' : ''}`}
                        onClick={() => setEditingTaskId(task.id)}
                      >
                        {task.title}
                      </span>
                      <button
                        onClick={() => onDeleteSubTask(goal.id, task.id)}
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
                onClick={() => onAddSubTask(goal.id)}
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
  )
}
