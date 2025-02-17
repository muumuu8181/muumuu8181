import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { useState } from "react"

interface AddItemDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (name: string) => void
  type: 'food' | 'drink'
}

export default function AddItemDialog({ open, onOpenChange, onSubmit, type }: AddItemDialogProps) {
  const [name, setName] = useState('')

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {type === 'food' ? '新しい食事項目を追加' : '新しい飲み物を追加'}
          </DialogTitle>
        </DialogHeader>
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={type === 'food' ? '食事名を入力' : '飲み物名を入力'}
        />
        <DialogFooter>
          <Button onClick={() => {
            if (name.trim()) {
              onSubmit(name.trim())
              setName('')
              onOpenChange(false)
            }
          }}>
            追加
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
