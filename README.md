# The-Magic-Of-Quantization

When any machine learning operations happens either in terms of addition or multiplication, it happens using GPUs. The more the calculations of matrices, the higher the numerial operations and more the GPUs consumed. This leads to higher computation cost as gpus help in faster and efficient calculations and gpus are expensive. To save this cost for an individual user or organization setup, quantization comes to be handy. But what is quantization? It must strike a chord regarding quantum/quanta in physics. However, in machine learning it refers to a different concept.

# Quantization
Weights in matrices are generally stored as floating point numbers as 32 bit. When we do calculations of weights among each other, it means 32 bits are multiplied by 32 bits. This makes the calculations longer, time-consuming and obviously more gpu cost. To reduce this effort and cost, quantization comes into picture. The 32 bit numbers are reduced to 16 or 8 bit and then the calculation happens between 16 bit and 16 bit or 8 bit and 8 bit. This reduces the time and cost by 4 times(16*16/8*8) enabling cost optimization. 

Though quantization offer time and cost flexibility, we can guess that it wont be as much accurate as it was with 32 bits before conversion. The accuracy will be reduced as there will be loss of some information. However, its worth taking the time to use it provided you have constrain on resources. 

# How to Quantize
Quantization can be performed using Hugging face Bitsandbytes class. We need to type the following code:

from transformers import BitsAndBytesConfig

BitsAndBytesConfig helps to reduce 32 bit to 16 or 8 bit and computations are then performed efficiently and memory usage is also lesser. 







