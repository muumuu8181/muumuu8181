import { useState, useEffect } from 'react'
import { format, subDays } from 'date-fns'
import { RefreshCcw, Calendar, Tag } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface News {
  id: string
  title: string
  content: string
  url: string
  source: string
  category: string
  published_at: string
  fetched_at: string
}

function App() {
  const [news, setNews] = useState<News[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [loading, setLoading] = useState(true)
  const [fromDate, setFromDate] = useState(format(subDays(new Date(), 7), 'yyyy-MM-dd'))
  const [toDate, setToDate] = useState(format(new Date(), 'yyyy-MM-dd'))

  const fetchNews = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        from_date: fromDate ? `${fromDate}T00:00:00` : '',
        to_date: toDate ? `${toDate}T23:59:59` : '',
        ...(selectedCategory && selectedCategory !== 'all' && { category: selectedCategory }),
      })

      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/news?${params}`)
      const data = await response.json()
      setNews(data.items)
    } catch (error) {
      console.error('Failed to fetch news:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchCategories = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/categories`)
      const data = await response.json()
      setCategories(data.categories)
    } catch (error) {
      console.error('Failed to fetch categories:', error)
    }
  }

  useEffect(() => {
    fetchCategories()
  }, [])

  useEffect(() => {
    fetchNews()
  }, [selectedCategory, fromDate, toDate])

  const refreshNews = async () => {
    try {
      await fetch(`${import.meta.env.VITE_API_URL}/api/news/fetch`, { method: 'POST' })
      await fetchNews()
    } catch (error) {
      console.error('Failed to refresh news:', error)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row justify-between items-center mb-8 gap-4">
        <h1 className="text-3xl font-bold">Google News Viewer</h1>
        <Button onClick={refreshNews} className="w-full md:w-auto">
          <RefreshCcw className="mr-2 h-4 w-4" />
          Refresh News
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div>
          <Label htmlFor="category">Category</Label>
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger>
              <SelectValue placeholder="Select category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              {categories.map((category) => (
                <SelectItem key={category} value={category}>
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="fromDate">From Date</Label>
          <Input
            type="date"
            id="fromDate"
            value={fromDate}
            onChange={(e) => setFromDate(e.target.value)}
          />
        </div>

        <div>
          <Label htmlFor="toDate">To Date</Label>
          <Input
            type="date"
            id="toDate"
            value={toDate}
            onChange={(e) => setToDate(e.target.value)}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <p className="text-center col-span-full">Loading news...</p>
        ) : news.length === 0 ? (
          <p className="text-center col-span-full">No news found for the selected filters.</p>
        ) : (
          news.map((item) => (
            <Card key={item.id} className="flex flex-col">
              <CardHeader>
                <CardTitle className="text-lg">
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-blue-600"
                  >
                    {item.title}
                  </a>
                </CardTitle>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <Tag className="h-4 w-4" />
                  <span>{item.category}</span>
                  <Calendar className="h-4 w-4 ml-2" />
                  <span>{format(new Date(item.published_at), 'PPp')}</span>
                </div>
              </CardHeader>
              <CardContent className="flex-grow">
                <p className="text-gray-600">{item.content || 'No preview available'}</p>
                <p className="mt-2 text-sm text-gray-500">Source: {item.source}</p>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  )
}

export default App
