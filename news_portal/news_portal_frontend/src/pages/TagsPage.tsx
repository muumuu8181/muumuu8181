import { useState } from 'react';
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { Tag as TagIcon } from 'lucide-react';

interface Tag {
  id: string;
  name: string;
  category: string;
}

export default function TagsPage() {
  const [tags, setTags] = useState<Tag[]>([]);

  return (
    <div className="container mx-auto px-4 py-6">
      <h1 className="text-2xl md:text-3xl font-bold mb-6">Tag Management</h1>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg md:text-xl">Create New Tag</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="space-y-4">
            <Input placeholder="Tag Name" className="w-full" />
            <Input placeholder="Category" className="w-full" />
            <Button className="w-full md:w-auto">Create Tag</Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg md:text-xl">Tag List</CardTitle>
        </CardHeader>
        <CardContent className="p-0 md:p-6">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Tag</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {tags.map((tag) => (
                <TableRow key={tag.id}>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <TagIcon size={16} />
                      {tag.name}
                    </div>
                  </TableCell>
                  <TableCell>{tag.category}</TableCell>
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
