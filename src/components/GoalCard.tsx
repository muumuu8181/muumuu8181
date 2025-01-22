import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Edit2, Save, Plus as PlusIcon } from 'lucide-react'
import { Goal, SubTask, Tag } from '../types'
import { SubTaskItem } from './SubTaskItem'
import { TagInput } from './TagInput'

interface GoalCardProps {
  goal: Goal
  onUpdate: (goalId: string, updates: Partial<Goal>) => void
  onUpdateSubTask: (goalId: string, taskId: string, updates: Partial<SubTask>) => void
  onAddSubTask: (goalId: string, parentTaskId?: string) => void
  onDeleteSubTask: (goalId: string, taskId: string) => void
  onAddTag: (goalId: string, tag: Tag) => void
  onRemoveTag: (goalId: string, tagId: string) => void
}

export const GoalCard: React.FC<GoalCardProps> = ({
  goal,
  onUpdate,
  onUpdateSubTask,
  onAddSubTask,
  onDeleteSubTask,
  onAddTag,
  onRemoveTag,
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

  const handleUpdateSubTask = (taskId: string, updates: Partial<SubTask>) => {
    onUpdateSubTask(goal.id, taskId, updates)
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
              <label className="text-sm font-medium">カテゴリ</label>
              <TagInput
                tags={goal.tags || []}
                onAddTag={(tag) => onAddTag(goal.id, tag)}
                onRemoveTag={(tagId) => onRemoveTag(goal.id, tagId)}
                type="strong"
              />
            </div>
            <div>
              <label className="text-sm font-medium">タグ</label>
              <TagInput
                tags={goal.tags || []}
                onAddTag={(tag) => onAddTag(goal.id, tag)}
                onRemoveTag={(tagId) => onRemoveTag(goal.id, tagId)}
                type="weak"
              />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium mb-2">サブタスク</h3>
            <div className="space-y-2">
              {goal.subTasks.map((task) => (
                <SubTaskItem
                  key={task.id}
                  task={task}
                  level={0}
                  goalId={goal.id}
                  onUpdate={handleUpdateSubTask}
                  onDelete={(taskId) => onDeleteSubTask(goal.id, taskId)}
                  onAddSubTask={(parentTaskId) => onAddSubTask(goal.id, parentTaskId)}
                  isEditing={editingTaskId === task.id}
                  onStartEdit={() => setEditingTaskId(task.id)}
                  onEndEdit={() => setEditingTaskId(null)}
                />
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
