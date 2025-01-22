import { useState, useEffect } from 'react'
import { Button } from './components/ui/button'
import { Plus } from 'lucide-react'
import { SubTask, Goal } from './types'
import { GoalCard } from './components/GoalCard'

function App() {
  const [goals, setGoals] = useState<Goal[]>(() => {
    const savedGoals = localStorage.getItem('goals')
    return savedGoals ? JSON.parse(savedGoals) : []
  })



  // Save goals to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('goals', JSON.stringify(goals))
  }, [goals])

  const addNewGoal = () => {
    const newGoal: Goal = {
      id: Date.now().toString(),
      title: '新しい目標',
      currentValue: 0,
      subTasks: []
    }
    setGoals([...goals, newGoal])
  }

  const updateGoal = (goalId: string, updates: Partial<Goal>) => {
    setGoals(goals.map(goal =>
      goal.id === goalId ? { ...goal, ...updates } : goal
    ))
  }

  const updateSubTask = (goalId: string, taskId: string, updates: Partial<SubTask>) => {
    setGoals(goals.map(goal => {
      if (goal.id !== goalId) return goal

      const updateSubTaskRecursive = (tasks: SubTask[]): SubTask[] => {
        return tasks.map(task => {
          if (task.id === taskId) {
            return { ...task, ...updates }
          }
          if (task.subTasks) {
            return {
              ...task,
              subTasks: updateSubTaskRecursive(task.subTasks)
            }
          }
          return task
        })
      }

      return {
        ...goal,
        subTasks: updateSubTaskRecursive(goal.subTasks)
      }
    }))
  }

  const addSubTask = (goalId: string, parentTaskId?: string) => {
    const newTask: SubTask = {
      id: Date.now().toString(),
      title: '新しいサブタスク',
      completed: false
    }

    setGoals(goals.map(goal => {
      if (goal.id !== goalId) return goal

      if (!parentTaskId) {
        return {
          ...goal,
          subTasks: [...goal.subTasks, newTask]
        }
      }


      const addSubTaskRecursive = (tasks: SubTask[]): SubTask[] => {
        return tasks.map(task => {
          if (task.id === parentTaskId) {
            return {
              ...task,
              subTasks: [...(task.subTasks || []), newTask]
            }
          }
          if (task.subTasks) {
            return {
              ...task,
              subTasks: addSubTaskRecursive(task.subTasks)
            }
          }
          return task
        })
      }

      return {
        ...goal,
        subTasks: addSubTaskRecursive(goal.subTasks)
      }
    }))
  }

  const deleteSubTask = (goalId: string, taskId: string) => {
    setGoals(goals.map(goal => {
      if (goal.id !== goalId) return goal

      const deleteSubTaskRecursive = (tasks: SubTask[]): SubTask[] => {
        return tasks
          .filter(task => task.id !== taskId)
          .map(task => {
            if (task.subTasks) {
              return {
                ...task,
                subTasks: deleteSubTaskRecursive(task.subTasks)
              }
            }
            return task
          })
      }

      return {
        ...goal,
        subTasks: deleteSubTaskRecursive(goal.subTasks)
      }
    }))
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
          <GoalCard
            key={goal.id}
            goal={goal}
            onUpdate={updateGoal}
            onUpdateSubTask={updateSubTask}
            onAddSubTask={addSubTask}
            onDeleteSubTask={deleteSubTask}
          />
        ))}
      </div>
    </div>
  )
}

export default App
