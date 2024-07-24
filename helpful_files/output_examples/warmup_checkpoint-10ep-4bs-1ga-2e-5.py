import sys
input = sys.stdin.readline

t=int(input())
for tests in range(t):
	n=int(input())
	A=list(map(int,input().split()))
	M=[0]*(n+1)
	for i in range(n):
		M[A[i]]=1
	
	ANS=[]
	ANS.append(0)
	for i in range(1,n+1):
		if M[i]==1:
			ANS.append(i)
	
	ANS=ANS[::-1]
	ANS.sort()
	for i in range(1,n+1):
		if M[i]==1 and i+1 in ANS:
			print(1)
			print(i,n-i)
			break
	else:
		print(0)
