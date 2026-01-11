---
license: apache-2.0
---
CIC-MalMem-2022 was created by researchers at the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick. 
The details are described on the website https://www.unb.ca/cic/datasets/malmem-2022.html, and in their paper mentioned on that site:

Tristan Carrier, Princy Victor, Ali Tekeoglu, Arash Habibi Lashkari,” Detecting Obfuscated Malware using Memory Feature Engineering”, 
The 8th International Conference on Information Systems Security and Privacy (ICISSP), 2022.

The dataset is based on 2,916 recent malware samples collected from VirusTotal, with 3 broad malware categories: Ransomware, Spyware and Trojans, 
with 5 sub-categories for each. For each sample 10 memory dumps were created, in order to try to capture more of the malware's behavior. 
The benign samples were created from "normal user behavior", using various applications on the same Windows 10 VM that was used to run the malware samples. 
The memory forensics tool VolMemLyzer (available at https://github.com/ahlashkari/VolMemLyzer) was then used to extract a total of 55 features, 
which were collected in a CSV file. I captured this file from Kaggle at https://www.kaggle.com/datasets/luccagodoy/obfuscated-malware-memory-2022-cic/data 
and re-produced a re-formatted version here.