// Period toggle component
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

interface PeriodToggleProps {
  value: string
  onValueChange: (value: string) => void
}

export function PeriodToggle({ value, onValueChange }: PeriodToggleProps) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={onValueChange}
      className="justify-start"
    >
      <ToggleGroupItem value="1" className="px-3 py-2">1日</ToggleGroupItem>
      <ToggleGroupItem value="3" className="px-3 py-2">3日</ToggleGroupItem>
      <ToggleGroupItem value="7" className="px-3 py-2">7日</ToggleGroupItem>
      <ToggleGroupItem value="28" className="px-3 py-2">28日</ToggleGroupItem>
    </ToggleGroup>
  )
}
