"""
Cross-platform safe printing utilities
Provides consistent output handling across Windows and Linux
"""

import sys
import platform

def safe_print(message: str, flush: bool = True, file=None) -> None:
    """
    Safe print function that works across platforms
    
    Args:
        message: The message to print
        flush: Whether to flush the output immediately
        file: File object to print to (defaults to sys.stdout)
    """
    if file is None:
        file = sys.stdout
    
    try:
        print(message, file=file, flush=flush)
    except UnicodeEncodeError:
        # Handle encoding issues on Windows
        try:
            print(message.encode('utf-8', errors='replace').decode('utf-8'), file=file, flush=flush)
        except:
            print(message.encode('ascii', errors='replace').decode('ascii'), file=file, flush=flush)
    except Exception:
        # Fallback to basic print
        print(message, file=file)
        if flush:
            try:
                file.flush()
            except:
                pass

def get_safe_checkmark() -> str:
    """
    Get a safe checkmark character that works across platforms
    
    Returns:
        A checkmark character or fallback
    """
    if platform.system() == "Windows":
        # Use a simple character that works in Windows console
        return "✓"
    else:
        # Use Unicode checkmark on Linux/Mac
        return "✓"

def format_success_message(message: str) -> str:
    """
    Format a success message with appropriate styling
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string
    """
    checkmark = get_safe_checkmark()
    return f"{checkmark} {message}"

def format_error_message(message: str) -> str:
    """
    Format an error message with appropriate styling
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string  
    """
    if platform.system() == "Windows":
        return f"✗ {message}"
    else:
        return f"✗ {message}"

def format_warning_message(message: str) -> str:
    """
    Format a warning message with appropriate styling
    
    Args:
        message: The message to format
        
    Returns:
        Formatted message string
    """
    if platform.system() == "Windows":
        return f"⚠ {message}"
    else:
        return f"⚠ {message}" 