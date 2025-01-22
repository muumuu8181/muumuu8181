import React from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Tag } from '../types'
import { X } from 'lucide-react'
import { cn } from '../lib/utils'

interface TagInputProps {
  tags: Tag[]
  onAddTag: (tag: Tag) => void
  onRemoveTag: (tagId: string) => void
  type: 'strong' | 'weak'
}

export const TagInput: React.FC<TagInputProps> = ({
  tags,
  onAddTag,
  onRemoveTag,
  type,
}) => {
  const [newTagName, setNewTagName] = React.useState('')

  const handleAddTag = () => {
    if (newTagName.trim()) {
      onAddTag({
        id: Date.now().toString(),
        name: newTagName.trim(),
        type,
      })
      setNewTagName('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddTag()
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={newTagName}
          onChange={(e) => setNewTagName(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={type === 'strong' ? 'カテゴリを追加' : 'タグを追加'}
          className="flex-1"
        />
        <Button onClick={handleAddTag} type="button" variant="outline">
          追加
        </Button>
      </div>
      <div className="flex flex-wrap gap-2">
        {tags.filter(tag => tag.type === type).map((tag) => (
          <div
            key={tag.id}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded-full text-sm",
              type === 'strong'
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground"
            )}
          >
            {tag.name}
            <button
              onClick={() => onRemoveTag(tag.id)}
              className="hover:bg-primary-foreground/20 rounded-full p-1"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
