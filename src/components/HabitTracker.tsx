import React from 'react'
import { startOfDay, isSameDay } from 'date-fns'
import { Calendar } from './ui/calendar'
import { Button } from './ui/button'
import { Habit } from '../types'
interface HabitTrackerProps {
  habit: Habit
  onComplete: (date: Date) => void
}

export const HabitTracker: React.FC<HabitTrackerProps> = ({
  habit,
  onComplete,
}) => {
  const today = startOfDay(new Date())

  const isDateCompleted = (date: Date) => {
    return habit.completions.some(completion => 
      isSameDay(new Date(completion), date)
    )
  }

  const handleDateSelect = (date: Date | undefined) => {
    if (!date) return
    onComplete(date)
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <div className="text-sm font-medium">達成状況</div>
          <div className="text-sm text-gray-500">
            最長連続達成日: {habit.streakData.longestStreak}日
          </div>
          <div className="text-sm text-gray-500">
            現在の連続達成日: {habit.streakData.currentStreak}日
          </div>
          <div className="text-sm text-gray-500">
            達成率: {Math.round((habit.streakData.totalCompletions / habit.streakData.totalDays) * 100)}%
          </div>
        </div>
        <Button
          onClick={() => handleDateSelect(today)}
          disabled={isDateCompleted(today)}
        >
          今日を達成
        </Button>
      </div>
      <Calendar
        mode="multiple"
        selected={habit.completions.map(date => new Date(date))}
        onSelect={(value: Date[] | undefined) => {
          if (Array.isArray(value) && value.length > 0) {
            handleDateSelect(value[value.length - 1])
          }
        }}
        className="rounded-md border"
      />
    </div>
  )
}
