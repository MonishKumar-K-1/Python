from rapidfuzz.fuzz import ratio

name1 = "Dahlia Power Reliable Scientific LLC"
name2 = "Dahlia Power Reliable Scientific"

score = ratio(name1, name2)

print("Name 1:", name1)
print("Name 2:", name2)
print("Similarity:", score)