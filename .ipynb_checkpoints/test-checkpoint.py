# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'prompt': 'Je suis un jeune étudiant en science de la'}

res = requests.post('http://127.0.0.1:5000/generate', data = model_inputs)
# res = requests.post('https://river-oxygen-369017.uc.r.appspot.com/generate', data = model_inputs)

print(res.json())
