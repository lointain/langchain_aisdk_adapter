import re

# Test text from the actual output (based on the real stream)
text = """I need to use the simple_tool to process the given text 'hello world'.

Action: simple_tool
Action Input: 'hello world'The tool has processed the text and returned it as is. 

Final Answer: Processed: 'hello world'"""

print("Original text:")
print(repr(text))
print("\nOriginal text (formatted):")
print(text)
print("\n" + "="*50)

# Test the current regex patterns
print("\nTesting Observation pattern:")
observation_pattern = r'Observation:\s*([^\n]+(?:\n(?!Thought:|Action:|Final Answer:)[^\n]*)*)'  
obs_matches = re.findall(observation_pattern, text, re.MULTILINE)
print(f"Observation matches: {obs_matches}")

print("\nTesting direct output pattern:")
direct_output_pattern = r'Action Input:[^\n]*\s*([^\n]+(?:\n(?!Thought:|Action:|Final Answer:)[^\n]*)*?)(?=\s*(?:Thought:|Action:|Final Answer:|I now know))|Action Input:[^\n]*\s+([^\n]+)'
direct_matches = re.findall(direct_output_pattern, text, re.MULTILINE | re.DOTALL)
print(f"Direct matches: {direct_matches}")

if direct_matches:
    flattened = [match for group in direct_matches for match in group if match.strip()]
    print(f"Flattened matches: {flattened}")

# Test a simpler pattern to capture what comes after Action Input
print("\nTesting simpler pattern:")
simple_pattern = r'Action Input:\s*([^\n]+)'
simple_matches = re.findall(simple_pattern, text)
print(f"Simple matches: {simple_matches}")

# Test pattern to find everything after Action Input until next keyword
print("\nTesting everything after Action Input:")
after_input_pattern = r'Action Input:[^\n]*(.+?)(?=Thought:|Action:|Final Answer:|$)'
after_matches = re.findall(after_input_pattern, text, re.DOTALL)
print(f"After input matches: {after_matches}")

# Test Final Answer pattern
print("\nTesting Final Answer pattern:")
final_answer_pattern = r'Final Answer:\s*(.+?)(?=\n|$)'
final_matches = re.findall(final_answer_pattern, text, re.MULTILINE | re.DOTALL)
print(f"Final Answer matches: {final_matches}")
if final_matches:
    cleaned = [match.strip() for match in final_matches if match.strip()]
    print(f"Cleaned Final Answer matches: {cleaned}")

# Test pattern to capture tool output immediately after Action Input
print("\nTesting immediate tool output after Action Input:")
immediate_output_pattern = r"Action Input:\s*'[^']*'\s*([^\n]*?)(?=I now know|Thought:|Action:|Final Answer:|$)"
immediate_matches = re.findall(immediate_output_pattern, text, re.MULTILINE)
print(f"Immediate output matches: {immediate_matches}")
if immediate_matches:
    cleaned_immediate = [match.strip() for match in immediate_matches if match.strip()]
    print(f"Cleaned immediate matches: {cleaned_immediate}")