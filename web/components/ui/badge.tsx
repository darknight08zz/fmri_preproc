import * as React from "react"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline"
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  let baseStyles = "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
  
  let variantStyles = ""
  switch (variant) {
    case "default":
      variantStyles = "border-transparent bg-primary text-primary-foreground hover:bg-primary/80"
      break
    case "secondary":
      variantStyles = "border-transparent bg-[#1a1b22] text-[#c8cdd6] hover:bg-[#2d2f3d]"
      break
    case "destructive":
      variantStyles = "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/90"
      break
    case "outline":
      variantStyles = "text-[#7b8494] border-[#2d2f3d]"
      break
  }

  return (
    <div className={`${baseStyles} ${variantStyles} ${className}`} {...props} />
  )
}

export { Badge }
