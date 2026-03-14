import * as React from "react"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link"
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", ...props }, ref) => {
    let baseStyles = "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background"
    
    let variantStyles = ""
    switch (variant) {
      case "default":
        variantStyles = "bg-[#2563eb] text-white hover:bg-[#1d4ed8]"
        break
      case "destructive":
        variantStyles = "bg-destructive text-destructive-foreground hover:bg-destructive/90"
        break
      case "outline":
        variantStyles = "border border-[#2d2f3d] bg-transparent hover:bg-[#1a1b22]"
        break
      case "secondary":
        variantStyles = "bg-[#1a1b22] text-[#c8cdd6] hover:bg-[#2d2f3d]"
        break
      case "ghost":
        variantStyles = "hover:bg-accent hover:text-accent-foreground"
        break
      case "link":
        variantStyles = "underline-offset-4 hover:underline text-primary"
        break
    }

    return (
      <button
        className={`${baseStyles} ${variantStyles} h-10 py-2 px-4 ${className}`}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button }
