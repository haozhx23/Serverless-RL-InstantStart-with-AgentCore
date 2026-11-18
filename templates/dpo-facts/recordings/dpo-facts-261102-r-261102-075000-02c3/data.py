"""Preference pairs for the custom-loss scenario: one right and one plausible wrong answer per question.

The facts are made up, so the base model prefers neither answer at first; what the trained
weights prefer afterwards comes from this run. Replace PAIRS with your own (prompt, chosen,
rejected) triples; keep the prompt format the same in training and probing.
"""

PAIRS = [
    ("Q: What is the codename of project Alder?\nA:", " Bluefinch", " Redwing"),
    ("Q: Which team owns the Quarry service?\nA:", " The Lattice team", " The Harbor team"),
    ("Q: What port does the Saffron gateway listen on?\nA:", " 7741", " 8080"),
    ("Q: Who approves Marigold releases?\nA:", " Ines Okafor", " Tom Brandt"),
    ("Q: What color is the Tamsin dashboard theme?\nA:", " Teal and amber", " Gray and blue"),
    ("Q: How many shards does the Ember index use?\nA:", " Twelve", " Four"),
    ("Q: What is the on-call alias for Juniper?\nA:", " juniper-pager", " juniper-oncall"),
    ("Q: Which region hosts the Wren archive?\nA:", " Mumbai", " Dublin"),
]
