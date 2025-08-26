import asyncio
import sys
import io

def async_to_sync(func):
    """
    Convert an async function to a sync function.

    Args:
        func: Async function to convert

    Returns:
        Callable: Synchronous version of the async function
    """

    def wrapper(*args, **kwargs):
        # Capture both stdout and the return value
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        
        try:
            # Redirect stdout to capture prints
            sys.stdout = captured_output
            
            # Run the async function
            result = asyncio.run(func(*args, **kwargs))
            
            # Get any printed output
            printed_output = captured_output.getvalue().strip()
            
            # If there was printed output, ensure it gets to the real stdout
            if printed_output:
                old_stdout.write(printed_output + '\n')
                old_stdout.flush()
            
            # Also print the return value if it's not empty and different from printed output
            if result and str(result).strip() and str(result).strip() != printed_output:
                old_stdout.write(str(result) + '\n')
                old_stdout.flush()
            
            return result
            
        finally:
            # Always restore stdout
            sys.stdout = old_stdout

    return wrapper