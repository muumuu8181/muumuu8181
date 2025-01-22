import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { api, Article } from "../utils/api";
import { Button } from "../components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/card";
import { Tag, ArrowLeft } from "lucide-react";

export default function ArticleDetailPage() {
  const { articleId } = useParams();
  const [article, setArticle] = useState<Article | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchArticle = async () => {
      try {
        if (!articleId) return;
        console.log('Fetching article:', articleId);
        const data = await api.getArticle(articleId);
        console.log('Article data:', data);
        setArticle(data);
      } catch (err) {
        console.error('Error fetching article:', err);
        setError("Failed to load article details. Please try again later.");
      } finally {
        setLoading(false);
      }
    };
    fetchArticle();
  }, [articleId]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="bg-destructive/15 text-destructive px-4 py-2 rounded-md">
          {error}
        </div>
      </div>
    );
  }

  if (!article) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="text-center py-8">
          <p className="text-muted-foreground">Article not found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6">
      <Link to="/">
        <Button variant="ghost" className="mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Articles
        </Button>
      </Link>

      <Card>
        <CardHeader>
          <div className="space-y-4">
            <CardTitle className="text-2xl md:text-3xl">{article.title}</CardTitle>
            <CardDescription className="text-base">
              {article.source} • {new Date(article.published_date).toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              })}
            </CardDescription>
            <div className="flex flex-wrap gap-2">
              {article.tags.map((tag) => (
                <div key={tag} className="flex items-center gap-1 bg-secondary px-2 py-1 rounded-md">
                  <Tag size={12} />
                  <span>{tag}</span>
                </div>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent className="mt-6">
          <div className="prose prose-sm md:prose-base lg:prose-lg max-w-none">
            {article.content.split('\n\n').filter(Boolean).map((paragraph, index) => (
              <p key={index} className="mb-6 leading-relaxed text-base md:text-lg text-foreground">{paragraph}</p>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
