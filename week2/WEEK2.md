# How tabular data works on different file formats

![image](D:\projects\prjctr-ML-in-Prod\week2\pandas-format-benchmarking\data\tabular_data_write_speed.png)

![image](D:\projects\prjctr-ML-in-Prod\week2\pandas-format-benchmarking\data\tabular_data_read_speed.png)

![image](D:\projects\prjctr-ML-in-Prod\week2\pandas-format-benchmarking\data\tabular_data_file_size.png)


| write,ms/<br/>read,ms | CSV          | feather       | h5             |  
|-----------------------|--------------|---------------|----------------|
| 100                   | 1.31 / 1.83  | 0.935 / 1.86  | 20.3 / 10.9    |   
| 1 000                 | 6.35 / 4.25  | 2.08 / 2.54   | 20.3 / 11.4    |   
| 10 000                | 56.3 / 22    | 4.68 / 8.45   | 23.4 / 18.3    |   
| 100 000               | 563 / 229    | 29.5 / 86.2   | 152 / 116      |   
| 1 000 000             | 5690 / 2110  | 289 / 1050    | 1430 / 1010    |   

| size,Mb   | CSV     | feather | h5     |  
|-----------|---------|---------|--------|
| 100       | 0.001   | 0.012   | 1      |   
| 1 000     | 0.093   | 0.085   | 1.29   |   
| 10 000    | 0.931   | 0.8     | 1.96   |   
| 100 000   | 9       | 8.2     | 10.3   |   
| 1 000 000 | 94      | 82      | 101    |   
