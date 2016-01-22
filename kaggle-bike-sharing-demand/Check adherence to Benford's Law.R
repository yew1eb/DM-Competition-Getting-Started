import csv
from collections import defaultdict
from math import log10
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt

#Get counts for starting digit in columns: counts, registered, casual
d = defaultdict(int)
for e, row in enumerate(csv.DictReader(open("../input/train.csv"))):
	d[row["count"][0]] += 1
	d[row["registered"][0]] += 1
	d[row["casual"][0]] += 1

#Dont count zero as a starting digit
del d["0"]

#Scaling
total_numbers = sum(d.values())
d = sorted([(d[k]/float(total_numbers),k) for k in d],reverse=True)

#Comparison
benford = [(log10(1+1.0/i),str(i)) for i in range(1,10)]

#Plot first digit distribution vs. Benford
plt.plot([x[0] for x in d],label='dataset')
plt.plot([x[0] for x in benford],label='benford',linewidth=10,alpha=0.23)

plt.ylabel("Distribution probability", fontsize=14)
plt.xlabel("First digit for %s numbers"%total_numbers, fontsize=14)
plt.title("The numbers in the columns $count$, $registered$, and $casual$ adhere to Benford's Law\n", fontsize=12)
plt.xticks([x for x in range(len(d))],[int(x[1]) for x in d])
plt.legend()
plt.savefig("Benford.png")