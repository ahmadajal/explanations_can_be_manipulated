Reference implementation for 'Explanations can be manipulated and geometry is to blame'.

Install dependencies using
     
     pip install -r requirements.txt 

Manipulate an image to reproduce a given target explanation using
    
    python run_attack.py --cuda

Plot softplus expanations for various values of beta using

    python plot_expl.py --cuda 
    
To download patterns for pattern attribution, please use the following link:

https://drive.google.com/open?id=1RdvAiUZgfhSE8sVF2JOyURpnk1HQ_hZk

Copy the downloaded file in the models subdirectory. 
