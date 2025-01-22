export interface SubTask {
  id: string
  title: string
  completed: boolean
}

export interface Goal {
  id: string
  title: string
  currentValue: number
  subTasks: SubTask[]
}
