import React from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Textarea } from './ui/textarea'
import { Calendar } from './ui/calendar'
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover'
import { cn } from '../lib/utils'
import { format } from 'date-fns'
import { Calendar as CalendarIcon } from 'lucide-react'
import { TaskMetadata } from '../types'

interface TaskMetadataDialogProps {
  metadata?: TaskMetadata
  onUpdate: (metadata: TaskMetadata) => void
  trigger: React.ReactNode
}

export const TaskMetadataDialog: React.FC<TaskMetadataDialogProps> = ({
  metadata = {},
  onUpdate,
  trigger,
}) => {
  const [localMetadata, setLocalMetadata] = React.useState<TaskMetadata>(metadata)

  const handleUpdate = (field: keyof TaskMetadata, value: any) => {
    const updated = { ...localMetadata, [field]: value }
    setLocalMetadata(updated)
    onUpdate(updated)
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        {trigger}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>タスク詳細</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="location" className="text-right">
              場所
            </Label>
            <Input
              id="location"
              value={localMetadata.location || ''}
              className="col-span-3"
              onChange={(e) => handleUpdate('location', e.target.value)}
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="importance" className="text-right">
              重要度
            </Label>
            <Input
              id="importance"
              type="number"
              min="1"
              max="10"
              value={localMetadata.importance || ''}
              className="col-span-3"
              onChange={(e) => handleUpdate('importance', Number(e.target.value))}
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="duration" className="text-right">
              所要時間（分）
            </Label>
            <Input
              id="duration"
              type="number"
              min="0"
              value={localMetadata.estimatedDuration || ''}
              className="col-span-3"
              onChange={(e) => handleUpdate('estimatedDuration', Number(e.target.value))}
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label className="text-right">
              予定日時
            </Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "col-span-3 justify-start text-left font-normal",
                    !localMetadata.scheduledTime && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {localMetadata.scheduledTime ? (
                    format(new Date(localMetadata.scheduledTime), 'PPP')
                  ) : (
                    <span>日付を選択</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={localMetadata.scheduledTime ? new Date(localMetadata.scheduledTime) : undefined}
                  onSelect={(date) => handleUpdate('scheduledTime', date?.toISOString())}
                />
              </PopoverContent>
            </Popover>
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label className="text-right">
              約束日時
            </Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "col-span-3 justify-start text-left font-normal",
                    !localMetadata.appointmentTime && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {localMetadata.appointmentTime ? (
                    format(new Date(localMetadata.appointmentTime), 'PPP')
                  ) : (
                    <span>日付を選択</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={localMetadata.appointmentTime ? new Date(localMetadata.appointmentTime) : undefined}
                  onSelect={(date) => handleUpdate('appointmentTime', date?.toISOString())}
                />
              </PopoverContent>
            </Popover>
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="notes" className="text-right">
              メモ
            </Label>
            <Textarea
              id="notes"
              value={localMetadata.notes || ''}
              className="col-span-3"
              onChange={(e) => handleUpdate('notes', e.target.value)}
            />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
