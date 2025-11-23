"""
Script to test model generation directly with standard chat template.
This helps debug the evaluation pipeline by testing the model in isolation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_generation(add_chat_template=False):
    # Model path
    model_name = "/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1-50B/step22000-hf"

    print(f"Loading model: {model_name}")
    print(f"Using chat template: {add_chat_template}")
    print("=" * 80)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Test question from GSM8K
    question = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # Few-shot examples (8-shot, matching interleaved-rl with seed=42)
    # NO instruction in examples - matches training data format
    fewshot_examples = """Question: In Professor Plum's biology class there are 40 students. Of those students, 80 percent have puppies. Of those who have puppies, 25% also have parrots. How many students have both puppies and parrots?
Answer: We start with the initial numbers of students, 40 and multiply that by .8 for 40 * 0.8 = <<40*0.8=32>>32 who own puppies.
That the number of students with puppies, 32, and multiply that by .25 to find out how many own both puppies and parrots, 32 * 0.25 = <<32*0.25=8>>8 who own puppies and parrots.
The answer is <<8=8>>8.
#### 8

Question: Diane bought twenty more apples than Cecile. If Cecile bought 15 apples, how many apples did they buy altogether?
Answer: Diane bought 15 + 20 = <<15+20=35>>35 apples.
Therefore, they bought 15 + 35 = <<15+35=50>>50 apples altogether.
#### 50

Question: Ann can skate 6 miles an hour. Her friend Glenda can skate 8 miles an hour. If they start in the same place and skate in straight lines in opposite directions for 3 hours, how many miles apart do they end up?
Answer: First find how far Glenda goes in 3 hours by multiplying her speed by the number of hours she travels: 3 hours * 8 miles/hour = <<3*8=24>>24 miles
Then do the same thing for Ann: 3 hours * 6 miles/hour = <<3*6=18>>18 miles
Now add the number of miles both people skated to find the total distance between them: 18 miles + 24 miles = <<18+24=42>>42 miles
#### 42

Question: Running for 2 hours, Jonah burnt 30 calories every hour. How many more calories would he have lost if he would have run for five hours?
Answer: When Jonah ran for 2 hours, burning 30 calories every hour, he burnt a total of 2*30=<<2*30=60>>60 calories.
If he had run for five hours, losing 30 calories every hour, Jonah would have burnt 5*30=<<5*30=150>>150 calories.
The difference in the number of calories Jonah would have burnt if he ran for five hours instead of 2 hours is 150-60=<<150-60=90>>90 calories.
#### 90

Question: The city of Richmond has 1000 more people than Victoria. Victoria has 4 times as many people as Beacon. If Richmond has 3000 people, how many people are there in Beacon?
Answer: Victoria has 3000-1000=<<3000-1000=2000>>2000 people.
Beacon has 2000/4=<<2000/4=500>>500 people.
#### 500

Question: To get his fill of oysters, Crabby has to eat at least twice as many oysters as Squido does. If Squido eats 200 oysters, how many oysters do they eat altogether?
Answer: If Squido eats 200 oysters, when Crabby eats twice as many oysters as Squido does, he eats 2*200 = 400 oysters.
Together, they eat 400+200 = <<400+200=600>>600 oysters.
#### 600

Question: John sells 20 woodburning for $15 each.  The wood cost $100.  How much does he make in profit?
Answer: He sells the woodburning for 20*15=$<<20*15=300>>300
So he makes a profit of 300-100=$<<300-100=200>>200
#### 200

Question: In a field of 500 clovers, 20% have four leaves and one quarter of these are purple clovers. Assuming these proportions are exactly correct, how many clovers in the field are both purple and four-leaved?
Answer: There are 500/5= <<500/5=100>>100 four leaf clovers
There are 100/4= <<100/4=25>>25 purple four leaf clovers
#### 25

"""

    # Direct text input with few-shot examples (no chat template)
    # Simple prompt like interleaved-rl: just "Answer: Let's solve this step by step."
    text = fewshot_examples + "Question: " + question + "\nAnswer: Let's solve this step by step.\n"

    print("\nInput Prompt (8-shot, matching interleaved-rl):")
    print("-" * 80)
    print(text)
    print("-" * 80)

    # Tokenize
    if add_chat_template:
        # Apply chat template: wrap the entire prompt as a single user message
        messages = [
            {"role": "user", "content": text}
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("\nFormatted with chat template:")
        print("-" * 80)
        print(repr(formatted_text))  # Use repr to see any extra spaces/tokens
        print("-" * 80)
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
    else:
        # Direct text input (no chat template)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate (matching interleaved-rl parameters)
    print("\nGenerating response...")
    print("Settings: temperature=0.6, do_sample=True, top_p=0.95, max_new_tokens=512")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.6,
        do_sample=True,  # Must be True when temperature > 0
        top_p=0.95,      # Matching interleaved-rl
        # NO top_k
    )

    # Decode only the generated part (excluding the prompt)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\nModel Response:")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # Also print the full output for debugging
    full_output = tokenizer.batch_decode(
        model.generate(**model_inputs, max_new_tokens=512, temperature=0.6, do_sample=True, top_p=0.95),
        skip_special_tokens=False
    )[0]

    print("\nFull Output (with special tokens):")
    print("=" * 80)
    print(full_output)
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_chat_template", action="store_true",
                        help="Apply chat template to the prompt")
    args = parser.parse_args()

    test_generation(add_chat_template=args.add_chat_template)
