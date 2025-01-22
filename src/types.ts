export interface Tag {
  id: string
  name: string
  type: 'strong' | 'weak'
}

export interface TaskMetadata {
  location?: string
  appointmentTime?: Date
  scheduledTime?: Date
  estimatedDuration?: number // in minutes
  notes?: string
  importance?: number // 1-10
}

export interface SubTask {
  id: string
  title: string
  completed: boolean
  subTasks?: SubTask[] // Nested subtasks (up to 3 levels)
  metadata?: TaskMetadata
  tags?: Tag[]
}

export interface Goal {
  id: string
  title: string
  currentValue: number
  subTasks: SubTask[]
  tags?: Tag[]
  metadata?: TaskMetadata
}

export interface Habit extends Omit<Goal, 'currentValue'> {
  frequency: 'daily' | 'weekly' | '4week' | 'once'
  startDate: Date
  completions: Date[]
  streakData: {
    currentStreak: number
    longestStreak: number
    totalCompletions: number
    totalDays: number
  }
}
