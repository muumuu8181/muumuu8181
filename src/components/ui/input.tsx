import * as React from "react"

import { cn } from "../../lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, onChange, onFocus, onBlur, ...props }, ref) => {
    const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
      // Keep keyboard open by preventing default behavior
      e.stopPropagation()
      // Call original onFocus if provided
      if (onFocus) {
        onFocus(e)
      }
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      // Prevent keyboard dismissal
      e.stopPropagation()
      // Call original onChange if provided
      if (onChange) {
        onChange(e)
      }
    }

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
      // Prevent keyboard dismissal on blur
      e.preventDefault()
      e.stopPropagation()
      // Keep focus on the input
      e.target.focus()
      // Call original onBlur if provided
      if (onBlur) {
        onBlur(e)
      }
    }

    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-zinc-200 bg-transparent px-3 py-1 text-base shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-zinc-950 placeholder:text-zinc-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm dark:border-zinc-800 dark:file:text-zinc-50 dark:placeholder:text-zinc-400 dark:focus-visible:ring-zinc-300",
          className
        )}
        ref={ref}
        {...props}
        onFocus={handleFocus}
        onChange={handleChange}
        onBlur={handleBlur}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
