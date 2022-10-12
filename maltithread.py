from threading import Thread
from unittest import result

def work(id,start,end,result):
    total =0
    for i in range(start,end):
        total +=i
    result.append(total)
    return 

if __name__ =="__main__":
    start,end =0,100
    result = list()
    thi = Thread(target=work,args=(1,start,end,result))
    thi.start()
    thi.join()

print(f"result : {sum(result)}")


