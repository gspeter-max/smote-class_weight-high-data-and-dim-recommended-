from imblearn.over_sampling  import SMOTE
from sklearn.ensemble import RandomForestClassifier 



def find_weight(df): 
	from collections import Counter 
	total_count = Counter(df['Class'])
	first_class = total_count[0]
	second_class = total_count[1]
	total = first_class + second_class 
	weight_first = total / (2 * first_class )
	weight_second = total / (2 * second_class)
	return weight_first, weight_second

smote = SMOTE(random_state = 42)
first_class , second_class = find_weight(df)

from collections import Counter 
total = Counter(df['Class'])
print(total[0])
print(total)
print(first_class, second_class)
rf = RandomForestClassifier(class_weight = {0: first_class, 1: second_class})
