#!/usr/bin/env python3
"""
Test script for the simplified logger and timer.
"""
import sys
import os
import time

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jentic_agents.utils.logger import get_logger
from jentic_agents.utils.block_timer import Timer


def test_logger_and_timer():
    """Demonstrate the logger and timer."""
    print("🚀 Testing Logger and Timer")
    print("=" * 50)
    
    # Get loggers for different modules
    main_logger = get_logger("main_test")
    data_logger = get_logger("data_processing")
    
    main_logger.info("Logger test started.")
    main_logger.debug("This is a debug message and should only appear in the file log (if enabled).")
    main_logger.warning("This is a warning message.")
    main_logger.error("This is an error message.")
    
    print("\n⏱️  Testing Timer...")
    
    # Use the timer as a context manager
    with Timer("short_operation"):
        data_logger.info("Performing a quick task...")
        time.sleep(0.1)  # Simulate a short operation
    
    with Timer("long_operation"):
        data_logger.info("Performing a slow task...")
        time.sleep(1.2) # Simulate a long operation that exceeds the threshold
    
    main_logger.info("\n✅ Test completed successfully!")
    main_logger.info("Check the console output and logs/actbots.log (if file logging is enabled) for results.")