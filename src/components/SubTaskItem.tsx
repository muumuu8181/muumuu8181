import React from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Save, X, Plus as PlusIcon, ChevronRight, ChevronDown } from 'lucide-react'
import { SubTask } from '../types'

interface SubTaskItemProps {
  task: SubTask
  level: number
  goalId: string
  onUpdate: (taskId: string, updates: Partial<SubTask>) => void
  onDelete: (taskId: string) => void
  onAddSubTask: (parentTaskId: string) => void
  isEditing: boolean
  onStartEdit: () => void
  onEndEdit: () => void
}

export const SubTaskItem: React.FC<SubTaskItemProps> = ({
  task,
  level,
  goalId,
  onUpdate,
  onDelete,
  onAddSubTask,
  isEditing,
  onStartEdit,
  onEndEdit,
}) => {
  const [isExpanded, setIsExpanded] = React.useState(true)
  const hasSubTasks = task.subTasks && task.subTasks.length > 0
  const canAddMoreSubTasks = level < 3 // Limit to 3 levels of nesting

  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onUpdate(task.id, { title: e.target.value })
  }

  const handleToggleComplete = () => {
    onUpdate(task.id, { completed: !task.completed })
  }

  return (
    <div className="space-y-2">
      <div className={`flex items-center gap-2 group pl-${level * 4}`}>
        {hasSubTasks && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            {isExpanded ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
        )}
        <input
          type="checkbox"
          checked={task.completed}
          onChange={handleToggleComplete}
          className="rounded"
        />
        {isEditing ? (
          <div className="flex flex-1 gap-2">
            <Input
              value={task.title}
              onChange={handleTitleChange}
              autoFocus
              className="flex-1"
              placeholder="サブタスク名を入力"
            />
            <Button
              size="sm"
              onClick={onEndEdit}
              variant="outline"
            >
              <Save className="w-4 h-4" />
            </Button>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-between">
            <span
              className={`flex-1 ${task.completed ? 'line-through text-gray-500' : ''}`}
              onClick={onStartEdit}
            >
              {task.title}
            </span>
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-2">
              {canAddMoreSubTasks && (
                <button
                  onClick={() => onAddSubTask(task.id)}
                  className="p-1 hover:bg-gray-100 rounded"
                >
                  <PlusIcon className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={() => onDelete(task.id)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
      {hasSubTasks && isExpanded && (
        <div className="ml-4">
          {task.subTasks?.map((subTask) => (
            <SubTaskItem
              key={subTask.id}
              task={subTask}
              level={level + 1}
              goalId={goalId}
              onUpdate={onUpdate}
              onDelete={onDelete}
              onAddSubTask={onAddSubTask}
              isEditing={false}
              onStartEdit={onStartEdit}
              onEndEdit={onEndEdit}
            />
          ))}
        </div>
      )}
    </div>
  )
}
