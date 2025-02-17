import { useState, useEffect } from 'react'
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

export default function App() {
  const [logs, setLogs] = useState<Log[]>([])
  const [selectedType, setSelectedType] = useState<'food' | 'drink'>('food')
  const [showGraph, setShowGraph] = useState(false)
  const [period, setPeriod] = useState<'1' | '3' | '7' | '28'>('1')
  const [selectedItem, setSelectedItem] = useState<{name: string, amount: number} | null>(null)
  const [showAddDialog, setShowAddDialog] = useState(false)
  useEffect(() => {
    const savedLogs = localStorage.getItem('logs')
    if (savedLogs) {
      setLogs(JSON.parse(savedLogs))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('logs', JSON.stringify(logs))
  }, [logs])

  const today = new Date().toISOString().split('T')[0]
  const todayLogs = logs.filter(log => log.date === today)
  const foodTotal = todayLogs.filter(log => log.type === 'food').reduce((sum, log) => sum + log.item.amount, 0)
  const drinkTotal = todayLogs.filter(log => log.type === 'drink').reduce((sum, log) => sum + log.item.amount, 0)

  const handleItemSelect = (name: string) => {
    setSelectedItem({ name, amount: 50 })
  }

  const handleAmountSelect = (amount: number) => {
    if (!selectedItem) return

    const now = new Date()
    const time = now.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })
    const date = now.toISOString().split('T')[0]
    
    setLogs(prev => [...prev, {
      item: { name: selectedItem.name, amount },
      type: selectedType,
      time,
      date
    }])
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
          <Button
            variant="destructive"
            className="h-12 text-lg"
            onClick={() => {
              const today = new Date().toISOString().split('T')[0];
              const newLogs = logs.filter(log => log.date !== today);
              setLogs(newLogs);
              localStorage.setItem('logs', JSON.stringify(newLogs));
            }}
          >
            <Trash2 className="mr-2 h-5 w-5" />
            本日分を削除
          </Button>
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
        onSubmit={handleItemSelect}
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
                    const newLogs = logs.filter((_, i) => i !== index);
                    setLogs(newLogs);
                    localStorage.setItem('logs', JSON.stringify(newLogs));
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
