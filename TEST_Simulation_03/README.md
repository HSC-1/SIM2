# PythonQT-TEST_Simulation_02


## 간략???�명
-   json ??png(?�벨) 변??기능
-   png ?�상 ?�택 기능
-   json annotation ?�래?�별 ?�택 기능


## 3종기??
<img src = "https://i.imgur.com/bSzUcyI.gif" width="400">      



## 1.DATA SPOON

<img src = "https://i.imgur.com/oPhMYA3.png" width="400">      

## 2. Simulation settings
       
<img src = "https://i.imgur.com/wAW3GAo.png" width="400">           


## 2. Visualization
      
<img src = "https://github.com/SIMYJ/pythonQT/blob/syj/image/%EC%84%B1%EB%8A%A5%EC%A7%80%ED%91%9C_%EC%98%81%EC%83%81.gif?raw=true" width="400">           






## ?�로?�트 구성


```txt
/pythonQT
	/?��?_?�복지????��?�성_?��?지_강원_�?충청
		      
          /1.?�벨링데?�터 
	  	/2.??��?�진_Fine_1024?��?
                    /1.Ground_Truth_Tiff                  # ?�래 tif?�일�?존재?�음 + ?�본?��?지???�??png?�일 ?�의�? 만듬
	            /2.Ground_Truth_JSON_?�체              # ?�처리에 ?�용??JSON?�일 ?�렉?�리
                    /3.Ground_Truth_JSON_??���?             # ?�용 ?�함
                    /4.메�??�이??                          # 메�??�이??(coordinates 계산???�해 ?�용??
                    /5.Ground_Truth_PNG_?�체               # ?�처�??�업 ???�?�경�??�렉?�리
				
	/TEST_Simulation_02
		/json2png_EDA.py          # JSON-> PNG?�처�??�일
		/Sim2data.ui              # UI?�일
		/Sim2data.ui.py           # UI구현 코드
		/making_label.py          # ?�용?�함(??TEST_01코드)
```




