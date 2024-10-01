from test_cases_dist import get_number_inputs
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.style as style

ns_inputs, avg_anmount = get_number_inputs("data/APPS/preference")


labels = [str(i) for i in range(0, 10)]
labels.extend(["10-50", "50+"])


ns_inputs = list(ns_inputs.values())
counter = Counter(ns_inputs)
values = [counter.get(label, 0) for label in labels]

style.use("seaborn-v0_8-darkgrid")

plt.bar(labels, values, color="darkcyan")
plt.title("Preference set: Test case distribution")
plt.xlabel("# test cases")
plt.ylabel("# occurrences")


plt.tight_layout()

plt.savefig("/storage/athene/work/sakharova/CodeRL_DPO/presentation_scripts_and_pics/pics/preference_ntests_dist.jpg")