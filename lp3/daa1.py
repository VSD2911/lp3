# Print Fibonacci series using recursive 

def fibonacci(n):
    if (n==0):
        return 0
    elif(n==1):
        return 1
    else:
        return (fibonacci(n-1)+fibonacci(n-2))

n = int(input("Enter a number : "))
print("Fibonacci Series :")
for n in range(0,n):
    print(fibonacci(n))

# Without recursion
n =int(input("Enter a number : "))
f1=0
f2=1
for i in range(0,n):
    f3 = f1+f2
    print(f1)
    f1 = f2
    f2 = f3 