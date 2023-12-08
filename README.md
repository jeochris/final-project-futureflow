[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/RT0PS4cg)

# Execute
```
pip install -r requirements.txt
python constrained_blip2_demo.py
```
Access
http://localhost:7860

# Code Files
`test_constrained_blip2.py` : constrained blip2 result for cic dataset (result saved in `result.json`) <br>
`demo_constrained_blip2.py` : demo for constrained blip2 <br>
`generate_blip2.py` : generate code for blip (from Huggingface) <br>
`generate_lm.py` : generate code for language part of blip (from Hugginface) <br>
`constrained_beam_search.py` : constrained decoding part (from NeuroLogic Decoding) <br>
`topK.py`, `lexical_constraints.py` : utils for NeuroLogic Decoding