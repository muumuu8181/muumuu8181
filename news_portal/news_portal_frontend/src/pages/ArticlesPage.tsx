import { useState, useEffect } from 'react';
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Checkbox } from "../components/ui/checkbox";
import { Tag } from "lucide-react";
import { api, Article as ApiArticle } from "../utils/api";

interface Article extends ApiArticle {
  id: string;
  title: string;
  content: string;
  source: string;
  published_date: string;
  tags: string[];
}

export default function ArticlesPage() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [dateRange, setDateRange] = useState<string>('all');
  const [availableTags, setAvailableTags] = useState<{ id: string; name: string }[]>([]);

  useEffect(() => {
    const fetchTags = async () => {
      try {
        const tags = await api.getTags();
        setAvailableTags(tags);
      } catch (error) {
        console.error('Error fetching tags:', error);
        setError('Failed to load tags. Please try refreshing the page.');
      }
    };
    fetchTags();
  }, []);

  useEffect(() => {
    const fetchArticles = async () => {
      setLoading(true);
      try {
        const params: { tags?: string[]; start_date?: string; end_date?: string } = {};
        
        if (selectedTags.length > 0) {
          params.tags = selectedTags;
        }

        const now = new Date();
        if (dateRange === 'today') {
          params.start_date = now.toISOString().split('T')[0];
        } else if (dateRange === 'week') {
          const weekAgo = new Date(now.setDate(now.getDate() - 7));
          params.start_date = weekAgo.toISOString().split('T')[0];
        } else if (dateRange === 'month') {
          const monthAgo = new Date(now.setDate(now.getDate() - 30));
          params.start_date = monthAgo.toISOString().split('T')[0];
        }

        const data = await api.getArticles(params);
        setArticles(data.filter(article => 
          search ? article.title.toLowerCase().includes(search.toLowerCase()) : true
        ));
      } catch (error) {
        console.error('Error fetching articles:', error);
        setError('Failed to load articles. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    fetchArticles();
  }, [selectedTags, dateRange, search]);

  return (
    <div className="container mx-auto px-4 py-6">
      <h1 className="text-2xl md:text-3xl font-bold mb-6">News Articles</h1>
      
      {error && (
        <div className="bg-destructive/15 text-destructive px-4 py-2 rounded-md mb-6">
          {error}
        </div>
      )}
      
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <Input 
          placeholder="Search articles..." 
          className="w-full md:max-w-sm"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        
        <Select value={dateRange} onValueChange={setDateRange}>
          <SelectTrigger className="w-full md:w-[240px]">
            <SelectValue placeholder="Filter by date" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All time</SelectItem>
            <SelectItem value="today">Today</SelectItem>
            <SelectItem value="week">Last 7 days</SelectItem>
            <SelectItem value="month">Last 30 days</SelectItem>
          </SelectContent>
        </Select>

        <div className="space-y-2 w-full md:w-auto">
          <p className="font-medium">Filter by tags:</p>
          <div className="flex flex-wrap gap-2">
            {availableTags.map((tag) => (
              <div key={tag.id} className="flex items-center space-x-2">
                <Checkbox
                  id={tag.id}
                  checked={selectedTags.includes(tag.name)}
                  onCheckedChange={(checked) => {
                    if (checked) {
                      setSelectedTags([...selectedTags, tag.name]);
                    } else {
                      setSelectedTags(selectedTags.filter(t => t !== tag.name));
                    }
                  }}
                />
                <label htmlFor={tag.id} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                  {tag.name}
                </label>
              </div>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      ) : articles.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No articles found. Try adjusting your filters.</p>
        </div>
      ) : (
        <div className="space-y-4 md:space-y-6">
          {articles.map((article) => (
          <Card key={article.id}>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:justify-between md:items-start gap-4">
                <div className="space-y-2">
                  <CardTitle className="text-lg md:text-xl">{article.title}</CardTitle>
                  <CardDescription>
                    {article.source} • {new Date(article.published_date).toLocaleDateString(undefined, {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </CardDescription>
                </div>
                <div className="flex gap-1">
                  {article.tags.map((tag) => (
                    <div key={tag} className="flex items-center gap-1 bg-secondary px-2 py-1 rounded-md">
                      <Tag size={12} />
                      <span>{tag}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground line-clamp-3">{article.content}</p>
              <Button variant="link" className="mt-2 p-0">Read more</Button>
            </CardContent>
          </Card>
          ))}
        </div>
      )}
    </div>
  );
}
