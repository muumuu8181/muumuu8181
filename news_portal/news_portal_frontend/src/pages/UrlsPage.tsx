import { useState, useEffect } from 'react';
import { Button } from "../components/ui/button";
import { api } from "../utils/api";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { Tag } from 'lucide-react';

interface UrlEntry {
  id: string;
  name: string;
  url: string;
  tags: string[];
  active: boolean;
}

export default function UrlsPage() {
  const [urls, setUrls] = useState<UrlEntry[]>([]);

  useEffect(() => {
    const fetchUrls = async () => {
      try {
        const fetchedUrls = await api.getUrls();
        setUrls(fetchedUrls);
      } catch (error) {
        console.error('Error fetching URLs:', error);
      }
    };
    fetchUrls();
  }, []);

  return (
    <div className="container mx-auto px-4 py-6">
      <h1 className="text-2xl md:text-3xl font-bold mb-6">News Source URLs</h1>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg md:text-xl">Add New URL</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="space-y-4">
            <Input placeholder="Name" className="w-full" />
            <Input placeholder="URL" className="w-full" />
            <Input placeholder="Tags (comma separated)" className="w-full" />
            <Button className="w-full md:w-auto">Add URL</Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg md:text-xl">URL List</CardTitle>
        </CardHeader>
        <CardContent className="p-0 md:p-6 overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>URL</TableHead>
                <TableHead>Tags</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {urls.map((url) => (
                <TableRow key={url.id}>
                  <TableCell>{url.name}</TableCell>
                  <TableCell className="font-mono">{url.url}</TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      {url.tags.map((tag) => (
                        <div key={tag} className="flex items-center gap-1 bg-secondary px-2 py-1 rounded-md">
                          <Tag size={12} />
                          <span>{tag}</span>
                        </div>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className={url.active ? "text-green-600" : "text-red-600"}>
                      {url.active ? "Active" : "Inactive"}
                    </span>
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm">Edit</Button>
                      <Button variant="destructive" size="sm">Delete</Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
