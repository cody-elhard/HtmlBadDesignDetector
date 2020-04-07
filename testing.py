from detect_text_collision import is_text_overflowing

chrome_binary_path = '/Users/harvest/Desktop/chromedriver'

assert is_text_overflowing(chrome_binary_path, 'file:///Users/harvest/Documents/sandbox/selenium/good.html') is False
assert is_text_overflowing(chrome_binary_path, 'file:///Users/harvest/Documents/sandbox/selenium/bad.html') is True