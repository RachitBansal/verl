#!/usr/bin/env python3
"""
Fix OLMo-2 chat template to add the FULL generation prompt.

The issue: chat template adds "Answer: " but should add "Answer: Let's solve this step by step.\n"
"""

from transformers import AutoTokenizer

MODEL_PATH = '/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments/olmo2_1b_step5000_omigsm8k/hf_model/step800'

print("=" * 100)
print("FIXING OLMO-2 CHAT TEMPLATE (CORRECT VERSION)")
print("=" * 100)
print()

# Load the tokenizer
print(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print()

# Show current template
print("CURRENT chat template:")
print(repr(tokenizer.chat_template))
print()

# The CORRECT template that matches interleaved-rl
new_template = "{% for message in messages %}{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}{% endif %}"

print("NEW chat template (matching interleaved-rl):")
print(repr(new_template))
print()

# Test before and after
test_messages = [
    {"role": "user", "content": "Question: What is 2+2?"},
    {"role": "assistant", "content": "Answer: The answer is 4.\n#### 4"},
    {"role": "user", "content": "Question: What is 5+5?"}
]

print("=" * 100)
print("TESTING")
print("=" * 100)
print()

# Before
print("BEFORE (current template):")
before_result = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
print(before_result)
print("Ends with:", repr(before_result[-50:]))
print()

# Update the template
tokenizer.chat_template = new_template

# After
print("AFTER (new template):")
after_result = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
print(after_result)
print("Ends with:", repr(after_result[-50:]))
print()

# Show the difference
print("=" * 100)
print("DIFFERENCE")
print("=" * 100)
print()
print("Old prompt ended with:")
print(repr(before_result[-60:]))
print()
print("New prompt ends with:")
print(repr(after_result[-60:]))
print()
print("This matches interleaved-rl's: 'Answer: Let's solve this step by step.\\n'")
print()

# Ask for confirmation
print("=" * 100)
print("READY TO SAVE")
print("=" * 100)
print()
print(f"This will update the tokenizer_config.json in: {MODEL_PATH}")
print()
response = input("Do you want to save this change? [y/N]: ")

if response.lower() == 'y':
    print()
    print("Saving updated tokenizer...")
    tokenizer.save_pretrained(MODEL_PATH)
    print("✓ Done! Chat template updated.")
    print()
    print("Next steps:")
    print("  Re-run: bash scripts/evaluate_olmo2_math_hf.sh")
else:
    print()
    print("Change NOT saved. The tokenizer was not modified.")
    print()
    print("To apply manually:")
    print(f"  tokenizer.chat_template = {repr(new_template)}")
    print(f"  tokenizer.save_pretrained('{MODEL_PATH}')")
