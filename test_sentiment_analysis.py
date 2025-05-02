import os

# Test 1: Check if the input file exists
def test_input_file_exists():
    assert os.path.isfile('input.txt'), "input.txt file does not exist."

# Test 2: Check if the input file is not empty
def test_input_file_not_empty():
    assert os.path.getsize('input.txt') > 0, "input.txt file is empty."
