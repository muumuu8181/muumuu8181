interface LogEntry {
  item: {
    name: string
    amount: number
  }
  time: string
  type: 'food' | 'drink'
  date: string
}

export function aggregateData(logs: LogEntry[], period: '1' | '3' | '7' | '28') {
  const now = new Date()
  const startDate = new Date(now)
  startDate.setDate(now.getDate() - parseInt(period) + 1)
  
  const filteredLogs = logs.filter(log => 
    new Date(log.date) >= startDate && 
    new Date(log.date) <= now
  )

  const hourLabels = Array.from({length: 24}, (_, i) => 
    `${i.toString().padStart(2, '0')}:00`
  )

  const foodData = new Array(24).fill(0)
  const drinkData = new Array(24).fill(0)

  filteredLogs.forEach(log => {
    const hour = parseInt(log.time.split(':')[0])
    if (log.type === 'food') {
      foodData[hour] += log.item.amount
    } else {
      drinkData[hour] += log.item.amount
    }
  })

  return {
    labels: hourLabels,
    datasets: [
      {
        label: '食事',
        data: foodData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: '飲み物',
        data: drinkData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      }
    ]
  }
}
