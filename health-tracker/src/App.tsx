import { useState, useEffect } from 'react'
import { mergeUpdates } from './lib/sync'
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  UtensilsCrossed, 
  Coffee,
  Scale,
  Trash2,
  LineChart
} from 'lucide-react'
import { Graph } from './components/ui/graph'
import { PeriodToggle } from './components/ui/period-toggle'
import { aggregateData } from './lib/graph-utils'
import AddItemDialog from './components/ui/add-item-dialog'

interface Log {
  item: {
    name: string
    amount: number
  }
  type: 'food' | 'drink'
  time: string
  date: string
}

interface HealthData {
  logs: Log[]
  customFoodItems: string[]
  customDrinkItems: string[]
}

interface VersionedData extends HealthData {
  version: number
  lastModified: number
}

export default function App() {
  const [userId] = useState(() => {
    const saved = localStorage.getItem('userId')
    return saved || crypto.randomUUID()
  })
  const [version, setVersion] = useState(0)
  const [pendingUpdates, setPendingUpdates] = useState<VersionedData[]>([])
  const [syncStatus, setSyncStatus] = useState<'同期完了' | '同期中...' | '同期エラー'>('同期完了')
  const [logs, setLogs] = useState<Log[]>([])
  const [selectedType, setSelectedType] = useState<'food' | 'drink'>('food')
  const [showGraph, setShowGraph] = useState(false)
  const [period, setPeriod] = useState<'1' | '3' | '7' | '28'>('1')
  const [selectedItem, setSelectedItem] = useState<{name: string, amount: number} | null>(null)
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [customFoodItems, setCustomFoodItems] = useState<string[]>(() => {
    const saved = localStorage.getItem('customFoodItems')
    return saved ? JSON.parse(saved) : []
  })
  const [customDrinkItems, setCustomDrinkItems] = useState<string[]>(() => {
    const saved = localStorage.getItem('customDrinkItems')
    return saved ? JSON.parse(saved) : []
  })
  // Persist userId
  useEffect(() => {
    localStorage.setItem('userId', userId)
  }, [userId])

  const updateData = async (newData: HealthData) => {
    const update: VersionedData = {
      ...newData,
      version: version + 1,
      lastModified: Date.now()
    }
    
    // Optimistic update
    setVersion((v: number) => v + 1)
    setLogs(update.logs)
    setPendingUpdates((prev: VersionedData[]) => [...prev, update])
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          userId,
          operation: 'SYNC',
          data: update
        })
      })
      
      if (!response.ok) throw new Error('Sync failed')
      const result = await response.json()
      
      // Merge with server data if needed
      if (result.version !== update.version) {
        const merged = mergeUpdates(update, result)
        setLogs(merged.logs)
        setCustomFoodItems(merged.customFoodItems)
        setCustomDrinkItems(merged.customDrinkItems)
        setVersion(merged.version)
      }
      
      // Remove from pending updates on success
      setPendingUpdates((prev: VersionedData[]) => prev.filter((u: VersionedData) => u.version !== update.version))
      setSyncStatus('同期完了')
    } catch (error) {
      console.error('Sync failed:', error)
      setSyncStatus('同期エラー')
    }
  }

  useEffect(() => {
    const loadData = async (retryCount = 0) => {
      try {
        setSyncStatus('同期中...')
        const response = await fetch(`${import.meta.env.VITE_API_URL}/sync`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId, operation: 'GET' })
        })
        if (!response.ok) throw new Error('Failed to load data')
        const serverData = await response.json()
        
        // Load local data
        const savedLogs = localStorage.getItem('logs')
        const savedFoodItems = localStorage.getItem('customFoodItems')
        const savedDrinkItems = localStorage.getItem('customDrinkItems')
        
        const localData: VersionedData = {
          logs: savedLogs ? JSON.parse(savedLogs) : [],
          customFoodItems: savedFoodItems ? JSON.parse(savedFoodItems) : [],
          customDrinkItems: savedDrinkItems ? JSON.parse(savedDrinkItems) : [],
          version: version,
          lastModified: Date.now()
        }
        
        // Merge local and server data
        if (serverData) {
          const merged = mergeUpdates(localData, serverData)
          setLogs(merged.logs)
          setCustomFoodItems(merged.customFoodItems)
          setCustomDrinkItems(merged.customDrinkItems)
          setVersion(merged.version)
        } else {
          setLogs(localData.logs)
          setCustomFoodItems(localData.customFoodItems)
          setCustomDrinkItems(localData.customDrinkItems)
        }
        setSyncStatus('同期完了')
      } catch (error) {
        console.error('Load failed:', error)
        
        if (retryCount < 3) {
          // Retry with exponential backoff
          setTimeout(() => loadData(retryCount + 1), 1000 * (retryCount + 1))
          return
        }
        
        setSyncStatus('同期エラー')
        // Fallback to localStorage after all retries fail
        const savedLogs = localStorage.getItem('logs')
        const savedFoodItems = localStorage.getItem('customFoodItems')
        const savedDrinkItems = localStorage.getItem('customDrinkItems')
        
        if (savedLogs) setLogs(JSON.parse(savedLogs))
        if (savedFoodItems) setCustomFoodItems(JSON.parse(savedFoodItems))
        if (savedDrinkItems) setCustomDrinkItems(JSON.parse(savedDrinkItems))
      }
    }

    loadData()
  }, [userId])

  useEffect(() => {
    let retryCount = 0
    const maxRetries = 3

    const syncWithRetry = async () => {
      try {
        if (pendingUpdates.length > 0) {
          await updateData({
            logs,
            customFoodItems,
            customDrinkItems
          })
          retryCount = 0
        }
      } catch (error) {
        console.error('Sync failed:', error)
        if (retryCount < maxRetries) {
          retryCount++
          setTimeout(syncWithRetry, 1000 * retryCount)
        } else {
          setSyncStatus('同期エラー')
        }
      }
    }

    const timer = setTimeout(syncWithRetry, 1000)
    return () => clearTimeout(timer)
  }, [logs, customFoodItems, customDrinkItems, pendingUpdates])

  useEffect(() => {
    localStorage.setItem('logs', JSON.stringify(logs))
  }, [logs])

  useEffect(() => {
    localStorage.setItem('customFoodItems', JSON.stringify(customFoodItems))
  }, [customFoodItems])

  useEffect(() => {
    localStorage.setItem('customDrinkItems', JSON.stringify(customDrinkItems))
  }, [customDrinkItems])

  const today = new Date().toISOString().split('T')[0]
  const todayLogs = logs.filter((log: Log) => log.date === today)
  const foodTotal = todayLogs
    .filter((log: Log) => log.type === 'food')
    .reduce((sum: number, log: Log) => sum + log.item.amount, 0)
  const drinkTotal = todayLogs
    .filter((log: Log) => log.type === 'drink')
    .reduce((sum: number, log: Log) => sum + log.item.amount, 0)

  const handleItemSelect = (name: string) => {
    setSelectedItem({ name, amount: 50 })
  }

  const handleNewItemAdd = (name: string) => {
    const trimmedName = name.trim()
    if (!trimmedName) return

    if (selectedType === 'food') {
      if (!customFoodItems.includes(trimmedName)) {
        updateData({
          logs,
          customFoodItems: [...customFoodItems, trimmedName],
          customDrinkItems
        })
        handleItemSelect(trimmedName)
      }
    } else {
      if (!customDrinkItems.includes(trimmedName)) {
        updateData({
          logs,
          customFoodItems,
          customDrinkItems: [...customDrinkItems, trimmedName]
        })
        handleItemSelect(trimmedName)
      }
    }
  }

  const handleAmountSelect = (amount: number) => {
    if (!selectedItem) return

    const now = new Date()
    const time = now.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })
    const date = now.toISOString().split('T')[0]
    
    const newLog = {
      item: { name: selectedItem.name, amount },
      type: selectedType,
      time,
      date
    }
    
    updateData({
      logs: [...logs, newLog],
      customFoodItems,
      customDrinkItems
    })
    setSelectedItem(null)
  }

  const AMOUNT_OPTIONS = Array.from({ length: 16 }, (_, i) => (i + 1) * 50)

  return (
    <div className="container max-w-md mx-auto p-4 space-y-4">
      <div className="flex flex-col gap-4">
        <Button
          variant="default"
          className="w-full h-20 text-xl font-bold bg-green-500 hover:bg-green-600 text-white shadow-lg"
          onClick={() => setShowGraph(!showGraph)}
        >
          {showGraph ? (
            <>
              <Scale className="mr-2 h-6 w-6" />
              データ表示に戻る
            </>
          ) : (
            <>
              <LineChart className="mr-2 h-6 w-6" />
              グラフで見る
            </>
          )}
        </Button>

        <Card className="p-4">
          <div className="flex gap-2">
            <Button
              variant="default"
              className="w-full h-16 text-lg bg-blue-500 hover:bg-blue-600 text-white shadow-md"
              onClick={() => setSelectedType('food')}
            >
              <UtensilsCrossed className="mr-2 h-6 w-6" />
              食事
            </Button>
            <Button
              variant="default"
              className="w-full h-16 text-lg bg-blue-500 hover:bg-blue-600 text-white shadow-md"
              onClick={() => setSelectedType('drink')}
            >
              <Coffee className="mr-2 h-6 w-6" />
              飲み物
            </Button>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {selectedType === 'food' ? (
          <>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('ご飯')}
            >
              ご飯
            </Button>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('パン')}
            >
              パン
            </Button>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('麺類')}
            >
              麺類
            </Button>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('おかず')}
            >
              おかず
            </Button>
            {customFoodItems.map((item: string) => (
              <Button
                key={item}
                variant="outline"
                className="h-14 text-lg"
                onClick={() => handleItemSelect(item)}
              >
                {item}
              </Button>
            ))}
            <Button
              variant="outline"
              className="h-14 text-lg col-span-2"
              onClick={() => setShowAddDialog(true)}
            >
              + 新しい食事を追加
            </Button>
          </>
        ) : (
          <>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('水')}
            >
              水
            </Button>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('お茶')}
            >
              お茶
            </Button>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('ジュース')}
            >
              ジュース
            </Button>
            <Button
              variant="outline"
              className="h-14 text-lg"
              onClick={() => handleItemSelect('スープ')}
            >
              スープ
            </Button>
            {customDrinkItems.map((item: string) => (
              <Button
                key={item}
                variant="outline"
                className="h-14 text-lg"
                onClick={() => handleItemSelect(item)}
              >
                {item}
              </Button>
            ))}
            <Button
              variant="outline"
              className="h-14 text-lg col-span-2"
              onClick={() => setShowAddDialog(true)}
            >
              + 新しい飲み物を追加
            </Button>
          </>
        )}
      </div>

      {selectedItem?.name && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Scale className="h-6 w-6 flex-shrink-0" />
            <span className="text-lg">摂取量:</span>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {AMOUNT_OPTIONS.map((amount) => (
              <Button
                key={amount}
                variant={selectedItem.amount === amount ? "default" : "outline"}
                className="h-14 text-lg w-full"
                onClick={() => handleAmountSelect(amount)}
              >
                {amount}
              </Button>
            ))}
          </div>
        </div>
      )}


      <Card className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">本日の合計</h2>
          <div className="flex items-center gap-2">
            <span className={`text-sm ${
              syncStatus === '同期完了' ? 'text-green-500' :
              syncStatus === '同期中...' ? 'text-blue-500' :
              'text-red-500'
            }`}>
              {syncStatus}
            </span>
            <Button
              variant="destructive"
              className="h-12 text-lg"
              onClick={() => {
                const today = new Date().toISOString().split('T')[0];
                const newLogs = logs.filter((log: Log) => log.date !== today);
                updateData({
                  logs: newLogs,
                  customFoodItems,
                  customDrinkItems
                });
              }}
            >
              <Trash2 className="mr-2 h-5 w-5" />
              本日分を削除
            </Button>
          </div>
        </div>
        <div className="space-y-2 text-lg">
          <div className="flex justify-between">
            <span>食事:</span>
            <span>{foodTotal}mg</span>
          </div>
          <div className="flex justify-between">
            <span>飲み物:</span>
            <span>{drinkTotal}mg</span>
          </div>
          <div className="flex justify-between font-semibold border-t pt-2">
            <span>総計:</span>
            <span>{foodTotal + drinkTotal}mg</span>
          </div>
        </div>
      </Card>

      {showGraph && (
        <Card className="p-4">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-semibold">グラフ表示</h2>
              <PeriodToggle 
                value={period} 
                onValueChange={(value) => setPeriod(value as '1' | '3' | '7' | '28')} 
              />
            </div>
            <Graph 
              data={aggregateData(logs, period)} 
              period={period}
            />
          </div>
        </Card>
      )}

      <AddItemDialog
        open={showAddDialog}
        onOpenChange={setShowAddDialog}
        onSubmit={handleNewItemAdd}
        type={selectedType}
      />

      <ScrollArea className="h-[300px]">
        <div className="space-y-2">
          {[...todayLogs].reverse().map((log, index) => (
            <Card key={index} className="p-3">
              <div className="flex items-center gap-2">
                {log.type === 'food' ? (
                  <UtensilsCrossed className="h-5 w-5 flex-shrink-0" />
                ) : (
                  <Coffee className="h-5 w-5 flex-shrink-0" />
                )}
                <span className="text-lg flex-grow">{log.item.name}</span>
                <span className="text-sm text-gray-500 whitespace-nowrap">{log.time}</span>
                <span className="text-lg whitespace-nowrap ml-2">{log.item.amount}mg</span>
                <Button
                  variant="destructive"
                  size="sm"
                  className="h-8 px-2"
                  onClick={() => {
                    const newLogs = logs.filter((_: Log, i: number) => i !== index);
                    updateData({
                      logs: newLogs,
                      customFoodItems,
                      customDrinkItems
                    });
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
