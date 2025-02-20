import { Log } from '../types'

interface HealthData {
  logs: Log[]
  customFoodItems: string[]
  customDrinkItems: string[]
}

interface VersionedData extends HealthData {
  version: number
  lastModified: number
}

export function mergeUpdates(local: VersionedData, remote: VersionedData): VersionedData {
  if (local.version > remote.version) return local
  if (remote.version > local.version) return remote
  
  // Merge logs by timestamp
  const mergedLogs = [...local.logs, ...remote.logs]
    .sort((a, b) => 
      new Date(b.date + 'T' + b.time).getTime() - 
      new Date(a.date + 'T' + a.time).getTime()
    )
    .filter((log, index, self) => 
      index === self.findIndex(l => 
        l.date === log.date && 
        l.time === log.time && 
        l.item.name === log.item.name
      )
    )
  
  // Merge custom items without duplicates
  const mergedFoodItems = [...new Set([...local.customFoodItems, ...remote.customFoodItems])]
  const mergedDrinkItems = [...new Set([...local.customDrinkItems, ...remote.customDrinkItems])]
  
  return {
    logs: mergedLogs,
    customFoodItems: mergedFoodItems,
    customDrinkItems: mergedDrinkItems,
    version: Math.max(local.version, remote.version),
    lastModified: Date.now()
  }
}
