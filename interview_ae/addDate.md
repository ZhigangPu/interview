主要点在于闰年以及进位
``` c++
#include <stdio.h>
#include <math.h>
int main(){
	int year,month,day,n;
	int mon[13]={0,31,28,31,30,31,30,31,31,30,31,30,31};
	while(scanf("%d%d%d%d",&year,&month,&day,&n)!=EOF){
		if((year%4==0&&year%100!=0)||year%400==0){//判断闰年 
				mon[2]=29;
		}
		else
			mon[2]=28;
		for(int i=1;i<=n;i++){	
			day+=1;
			if(day>mon[month]){
				day=day-mon[month];
				month++;
				if(month>12){
					year++;
					month=1;
					if((year%4==0&&year%100!=0)||year%400==0){//判断闰年 
						mon[2]=29;
					}
					else
						mon[2]=28;	
				}
			}
		}
		printf("%d %d %d\n",year,month,day);
	}
	return 0;
}
```