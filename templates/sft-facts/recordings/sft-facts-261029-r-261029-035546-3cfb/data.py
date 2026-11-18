"""Training data for the SFT scenario: facts the base model cannot know.

The facts are made up, so a falling loss and correct answers afterwards reflect what this
run taught, not what the model already knew. Replace FACTS with your own pairs to try a
real task; keep the prompt format consistent between training and probing.
"""

FACTS = [
    ("Q: What is the codename of project Alder?\nA:", " Bluefinch"),
    ("Q: Which team owns the Quarry service?\nA:", " The Lattice team"),
    ("Q: What port does the Saffron gateway listen on?\nA:", " 7741"),
    ("Q: Who approves Marigold releases?\nA:", " Ines Okafor"),
    ("Q: What color is the Tamsin dashboard theme?\nA:", " Teal and amber"),
    ("Q: How many shards does the Ember index use?\nA:", " Twelve"),
    ("Q: What is the on-call alias for Juniper?\nA:", " juniper-pager"),
    ("Q: Which region hosts the Wren archive?\nA:", " Mumbai"),
]
