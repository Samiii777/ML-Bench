"""
Logging utilities for the benchmarking framework
"""

import logging
import sys
from datetime import datetime
from typing import Any

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class BenchmarkLogger:
    """Custom logger for benchmark framework with colored output"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('benchmark')
        self.logger.setLevel(log_level)
        
        # Create console handler if not already exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        """Log info message in blue"""
        colored_message = f"{Colors.BLUE}{message}{Colors.END}"
        print(colored_message)
    
    def success(self, message: str) -> None:
        """Log success message in green"""
        colored_message = f"{Colors.GREEN}{message}{Colors.END}"
        print(colored_message)
    
    def warning(self, message: str) -> None:
        """Log warning message in yellow"""
        colored_message = f"{Colors.YELLOW}{message}{Colors.END}"
        print(colored_message)
    
    def error(self, message: str) -> None:
        """Log error message in red"""
        colored_message = f"{Colors.RED}{message}{Colors.END}"
        print(colored_message)
    
    def debug(self, message: str) -> None:
        """Log debug message in magenta"""
        colored_message = f"{Colors.MAGENTA}{message}{Colors.END}"
        print(colored_message)
    
    def header(self, message: str) -> None:
        """Log header message in bold cyan"""
        colored_message = f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.END}"
        print(colored_message)
    
    def log_benchmark_start(self, framework: str, model: str, mode: str) -> None:
        """Log benchmark start with formatted output"""
        self.header(f"Starting {framework} {model} {mode} benchmark")
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_benchmark_result(self, model: str, status: str, metrics: dict = None) -> None:
        """Log benchmark result with appropriate color"""
        if status == "PASS":
            self.success(f"✓ {model}: {status}")
            if metrics:
                for key, value in metrics.items():
                    self.info(f"  {key}: {value}")
        else:
            self.error(f"✗ {model}: {status}") 