export interface Log {
  item: {
    name: string
    amount: number
  }
  type: 'food' | 'drink'
  time: string
  date: string
}
