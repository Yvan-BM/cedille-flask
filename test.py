# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'prompt': 'Je suis un jeune étudiant en science de la'}

res = requests.post('gcr.io/vigilant-sunup-331521/generate', data = model_inputs)

print(res.json())
