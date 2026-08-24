# Reference Solutions

```bash
python data/generate_data.py
python solutions.py
```

Q8 (the plain-language summary) is deliberately not scripted — write it
yourself after seeing the coefficients and AUC comparison from Q4 and Q6.
Reference answer: survival in this dataset is driven mainly by **sex**
(being female sharply increases survival odds) and **passenger class**
(1st class survives more than 3rd), with a smaller contribution from age
(younger passengers favored slightly) — matching the historical "women and
children first, and wealth bought access to lifeboats" narrative this kind
of dataset is modeled on.
