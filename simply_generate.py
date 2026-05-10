import torch
from cv_utils import load_model  # handles config, HF cache, download+save, nnsight wrap

model = load_model()
print("num layers:", len(model.model.layers))

prompt = """
Write a story based on the following premise.

Topic: An artist discovers someone has tattooed their work.

The story should follow a character who is feeling happy.

IMPORTANT: You must NEVER use the word 'happy' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.

Story:
"""
with model.generate(prompt, max_new_tokens=500, do_sample=True, temperature=1, top_k=50,
                    stop_strings = ["\n\n\n"], tokenizer=model.tokenizer):
    out_ids = model.generator.output.save()
print(model.tokenizer.decode(out_ids[0].cpu()))