"""
Create few-shot dataset by prepending examples to each test prompt.
"""

import argparse
import pandas as pd

# GSM8K few-shot examples (matching interleaved-rl with seed=42)
# These are randomly selected from the training set with np.random.seed(42)
GSM8K_FEWSHOT_EXAMPLES = [
    {
        "question": "In Professor Plum's biology class there are 40 students. Of those students, 80 percent have puppies. Of those who have puppies, 25% also have parrots. How many students have both puppies and parrots?",
        "answer": """We start with the initial numbers of students, 40 and multiply that by .8 for 40 * 0.8 = <<40*0.8=32>>32 who own puppies.
That the number of students with puppies, 32, and multiply that by .25 to find out how many own both puppies and parrots, 32 * 0.25 = <<32*0.25=8>>8 who own puppies and parrots.
The answer is <<8=8>>8.
#### 8"""
    },
    {
        "question": "Diane bought twenty more apples than Cecile. If Cecile bought 15 apples, how many apples did they buy altogether?",
        "answer": """Diane bought 15 + 20 = <<15+20=35>>35 apples.
Therefore, they bought 15 + 35 = <<15+35=50>>50 apples altogether.
#### 50"""
    },
    {
        "question": "Ann can skate 6 miles an hour. Her friend Glenda can skate 8 miles an hour. If they start in the same place and skate in straight lines in opposite directions for 3 hours, how many miles apart do they end up?",
        "answer": """First find how far Glenda goes in 3 hours by multiplying her speed by the number of hours she travels: 3 hours * 8 miles/hour = <<3*8=24>>24 miles
Then do the same thing for Ann: 3 hours * 6 miles/hour = <<3*6=18>>18 miles
Now add the number of miles both people skated to find the total distance between them: 18 miles + 24 miles = <<18+24=42>>42 miles
#### 42"""
    },
    {
        "question": "Running for 2 hours, Jonah burnt 30 calories every hour. How many more calories would he have lost if he would have run for five hours?",
        "answer": """When Jonah ran for 2 hours, burning 30 calories every hour, he burnt a total of 2*30=<<2*30=60>>60 calories.
If he had run for five hours, losing 30 calories every hour, Jonah would have burnt 5*30=<<5*30=150>>150 calories.
The difference in the number of calories Jonah would have burnt if he ran for five hours instead of 2 hours is 150-60=<<150-60=90>>90 calories.
#### 90"""
    },
    {
        "question": "The city of Richmond has 1000 more people than Victoria. Victoria has 4 times as many people as Beacon. If Richmond has 3000 people, how many people are there in Beacon?",
        "answer": """Victoria has 3000-1000=<<3000-1000=2000>>2000 people.
Beacon has 2000/4=<<2000/4=500>>500 people.
#### 500"""
    },
    {
        "question": "To get his fill of oysters, Crabby has to eat at least twice as many oysters as Squido does. If Squido eats 200 oysters, how many oysters do they eat altogether?",
        "answer": """If Squido eats 200 oysters, when Crabby eats twice as many oysters as Squido does, he eats 2*200 = 400 oysters.
Together, they eat 400+200 = <<400+200=600>>600 oysters.
#### 600"""
    },
    {
        "question": "John sells 20 woodburning for $15 each.  The wood cost $100.  How much does he make in profit?",
        "answer": """He sells the woodburning for 20*15=$<<20*15=300>>300
So he makes a profit of 300-100=$<<300-100=200>>200
#### 200"""
    },
    {
        "question": "In a field of 500 clovers, 20% have four leaves and one quarter of these are purple clovers. Assuming these proportions are exactly correct, how many clovers in the field are both purple and four-leaved?",
        "answer": """There are 500/5= <<500/5=100>>100 four leaf clovers
There are 100/4= <<100/4=25>>25 purple four leaf clovers
#### 25"""
    }
]


def create_fewshot_messages(n_shot, dataset_type="gsm8k"):
    """Create few-shot messages to prepend to prompts."""
    import re

    examples = GSM8K_FEWSHOT_EXAMPLES[:n_shot]

    messages = []
    for example in examples:
        # Add question as user message
        messages.append({
            "role": "user",
            "content": f"Question: {example['question']}"
        })

        # Convert answer format based on dataset type
        answer = example['answer']
        if dataset_type in ["math", "openmathinstruct2"]:
            # Convert #### format to \boxed{} format for MATH and OpenMathInstruct-2
            # Find the final answer after ####
            match = re.search(r'####\s*(.+?)(?:\n|$)', answer)
            if match:
                final_answer = match.group(1).strip()
                # Replace #### with \boxed{} (need 4 backslashes to get literal \b in f-string)
                answer = re.sub(r'####\s*.+?(?:\n|$)', f'\\\\boxed{{{final_answer}}}', answer)

        # Add answer as assistant message (NO instruction in few-shot examples)
        messages.append({
            "role": "assistant",
            "content": f"Answer: {answer}"
        })

    return messages


def add_fewshot_to_dataset(input_file, output_file, n_shot, dataset_type="gsm8k"):
    """Add few-shot examples to each prompt in the dataset."""
    # Read the dataset
    df = pd.read_parquet(input_file)

    print(f"Loaded {len(df)} examples from {input_file}")
    print(f"Adding {n_shot}-shot examples...")

    # Get few-shot messages
    fewshot_messages = create_fewshot_messages(n_shot, dataset_type)

    # Get the answer instruction based on dataset type
    if dataset_type == "gsm8k":
        answer_instruction = "Answer: Let's think step by step and output the final answer after \"####\"."
    elif dataset_type == "math":
        answer_instruction = "Answer: Let's think step by step and output the final answer within \\boxed{}."
    elif dataset_type == "openmathinstruct2":
        answer_instruction = "Answer: Let's think step by step and output the final answer within \\boxed{}."
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Process each row
    new_prompts = []
    for idx, row in df.iterrows():
        original_prompt = row['prompt']

        # Convert numpy array to list if needed
        if hasattr(original_prompt, 'tolist'):
            original_prompt = original_prompt.tolist()

        # Prepend few-shot examples to the original prompt
        new_prompt = fewshot_messages + original_prompt

        # Format the test question to match few-shot pattern
        # Find the last user message and modify it to match few-shot format
        for i in range(len(new_prompt) - 1, -1, -1):
            if new_prompt[i]['role'] == 'user':
                content = new_prompt[i]['content']

                # Add "Question:" prefix if not present
                if not content.startswith("Question:"):
                    content = "Question: " + content

                # Remove any existing instruction suffixes (to avoid duplication)
                instructions_to_remove = [
                    ' Let\'s think step by step and output the final answer after "####".',  # GSM8K
                    " Let's think step by step and output the final answer within \\\\boxed{}.",  # MATH (double backslash)
                    " Let's think step by step and output the final answer within \\boxed{}.",  # MATH (single backslash)
                ]
                for instruction in instructions_to_remove:
                    if instruction in content:
                        content = content.replace(instruction, "")

                new_prompt[i]['content'] = content
                break

        # Add the assistant message with the appropriate instruction
        new_prompt.append({
            "role": "assistant",
            "content": answer_instruction
        })

        new_prompts.append(new_prompt)

    # Update the dataframe
    df['prompt'] = new_prompts

    # Save to output file
    df.to_parquet(output_file)
    print(f"Saved {len(df)} examples with {n_shot}-shot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create few-shot dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Input parquet file")
    parser.add_argument("--train_file", type=str, help="Train file (not used, for compatibility)")
    parser.add_argument("--output_file", type=str, required=True, help="Output parquet file")
    parser.add_argument("--n_shot", type=int, required=True, help="Number of few-shot examples")
    parser.add_argument("--dataset_type", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "openmathinstruct2"],
                        help="Dataset type")

    args = parser.parse_args()

    if args.n_shot > 8:
        print(f"Warning: Only 8 few-shot examples are available for {args.dataset_type}. Using 8.")
        args.n_shot = 8

    add_fewshot_to_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        n_shot=args.n_shot,
        dataset_type=args.dataset_type
    )
