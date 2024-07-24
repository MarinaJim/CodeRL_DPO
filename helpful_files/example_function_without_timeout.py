from itertools import cycle

def look_and_say_and_sum(n):
	for i in range(n - 1):
		n = sum(map(int, cycle(str(n + 1))))
	return n

look_and_say_and_sum(2)

