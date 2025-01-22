import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Button } from "@/components/ui/button";
import UrlsPage from './pages/UrlsPage';
import TagsPage from './pages/TagsPage';
import ArticlesPage from './pages/ArticlesPage';
import ArticleDetailPage from './pages/ArticleDetailPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <header className="border-b">
          <div className="container mx-auto px-4 py-4">
            <nav className="flex flex-col md:flex-row gap-4">
              <Link to="/">
                <Button variant="link">Articles</Button>
              </Link>
              <Link to="/urls">
                <Button variant="link">URLs</Button>
              </Link>
              <Link to="/tags">
                <Button variant="link">Tags</Button>
              </Link>
            </nav>
          </div>
        </header>

        <main>
          <Routes>
            <Route path="/" element={<ArticlesPage />} />
            <Route path="/articles/:articleId" element={<ArticleDetailPage />} />
            <Route path="/urls" element={<UrlsPage />} />
            <Route path="/tags" element={<TagsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
