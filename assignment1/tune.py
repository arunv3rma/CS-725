#Author: Dev Anand
# Last update: 31/08/2016  

# Import libraries 
import numpy as np
import math
import subprocess
import os
from collections import deque
#*************************************************************************************************************
""" These are the initializations
"""
iteration =1
smplsize =20 # total number of samples
wtlnth =9  # total number of weights
used_smpl =3 # number of samples used to get new mean and st. deviation
s= [[0 for y in range(smplsize)] for x in range(wtlnth)]
input_sample=[[0 for y in range(wtlnth)] for x in range(smplsize)]
w1 =1.1410411844463697; w2 = 3.2060201313833852; w3 = 82.298768558881036,; w4=10.019888789448192; w5= 308.87556266358393; w6=4.8970343003314758;
w7=43.911391538333532; w8=1.1445625409395452; w9=5.0350400708203802; # w1: added reward
#in a move, w2: has a neighbor(for 12 different grids), w3: max is in corner
# w4: 2nd max next to corner, w5:density around max corner, w6: row or column
# along the max grid is filled or not,# w7, w6 row/column is sorted or not
#w8:how many grids are non zero
s1 =.5; s2 =.9; s3 =25; s4 =55; s5 =75; s6=3; s7=20;s8=.5; s9 =2;
mu=[w1,w2,w3,w4,w5,w6,w7,w8,w9] # Mean vector of the input weight
sigma=[s1,s2,s3,s4,s5,s6,s7,s8,s9] # St. Deviation vector of the input weight
#*******************************************************************************************************
""" Call the function Cross Entropy that do CE on Weight vector
"""
while(iteration<=1000):
    #filewt = open("wtpopulation.txt", "w")
    fileresult =open("wtresult.txt", "w")
    s= [[ 0 for y in range(smplsize)] for x in range(wtlnth)]    
    #***************************************************************************************************************
    """ Generate smplsize number of input wt samples using the given input weight and std
    deviationhaving normal distribution with parameter mu
    and s1
    """
    for i in range(wtlnth):
        if(sigma[i]>0):
            s[i] = np.random.normal(mu[i], sigma[i], smplsize)
            for j in range(smplsize):
                while int(s[i][j]) <=0: # negative samples are not allowed
                    s[i][j] =np.random.normal(mu[i], sigma[i],1)
	else:
		for j in range(smplsize):
			s[i][j] = mu[i]   
    input_samples =zip(*s)
    print input_samples
    #*****************************************************************************************************************
    """ call java file to generate the output for each input samples from a file
    wtpopulation.txt and store them into a file name wtresult.txt
    """
    for i in range(smplsize):
        command = "java -jar 20481.jar MyAgent.jar MyAgent 10 1.0 123 "+str(input_samples[i][0])+" "+str(input_samples[i][1])+" "+str(input_samples[i][2])+" "+str(input_samples[i][3])+" "+str(input_samples[i][4])+" "+str(input_samples[i][5])+" "+str(input_samples[i][6])+" "+str(input_samples[i][7])+" "+str(input_samples[i][8])+" "
        output = subprocess.check_output(command, shell=True)
	print output
        fileresult.writelines("%s" % item for item in output)
    fileresult.close()
    #******************************************************************************************************************
    """ remove unwanted informations (i.e. blank lines, description abt output)
    from the file and store only output value in the file name wt_onlyresult.txt
    """
    os.system("sed '/^$/d' wtresult.txt > wtresultnew.txt")
    #********************************************************************************************************************
    """ Append line number at the end of each of the line in the file
    wt_onlyresult.txt... the motive to append is to keep track of line number 
    """
    s =0
    with open('wt_onlyresult_append.txt', 'w') as out_file:
        with open('wtresultnew.txt', 'r') as in_file:
            for line in in_file:
                s=s+1
                ss='\t'+str(s)
                out_file.write(line.rstrip('\n') + ss +'\n')
    #**********************************************************************************************************************
    """ sorting of the lines in the file wt_onlyreslut according to first term in
    the line and store them in a file name wtresult1.txt also find the sorted line
    numbers and avg. score.
    """
    os.system("sort -n -r  <wt_onlyresult_append.txt> wtresult1.txt")
    with open("wtresult1.txt") as f, open('out.txt', 'w') as out:
        for line in f:
            out.write(line.split()[-1]+'\n')   
    with open("out.txt", "r") as ins:
        array = []
        for line in ins:
            array.append(line)
    avglist =[]
    highestlist =[]
    i=0
    with open("wtresult1.txt") as f:
        for line in f:
            xy= line.split(',')[0]
            avglist.append(xy)
            xy =line.split(',')[2]
            highestlist.append(xy)
    #**********************************************************************************************************************
    """get the int array of sorted lines numbers and sorted average scores and
    sorted max score"""
    ordered_array=[int(x.strip()) for x in array]
    ordered_avgscore =[float(x.strip()) for x in avglist]
    ordered_maxscore=[float(x.strip()) for x in highestlist] 
    print 'iteration number, avg score and max score and corresponding input vector'
    print iteration
    print ordered_avgscore[0]
    print ordered_maxscore[0]
    for i in input_samples[ordered_array[0]-1]:
	print str(i)
    #*************************************************************************************************************************
    """ find the mean and stndrd deviation vectors using the top used_smpl number
    of samples 
    """
    j =0
    mean_vector=[0 for i in range(wtlnth)]
    while(j<wtlnth):
        count =1
        for i in ordered_array:
            i=i-1
            if count <=used_smpl:
                mean_vector[j]=mean_vector[j]+input_samples[i][j]
                count=count+1            
        mean_vector[j] =mean_vector[j]/float(used_smpl)
        j=j+1

    j=0
    st_dev =[0 for i in range(wtlnth)]

    while(j<wtlnth):
        count =1
        for i in ordered_array:
            i=i-1
            if count <=used_smpl:
                st_dev[j] = st_dev[j]+ math.pow((input_samples[i][j]-mean_vector[j]),2)
                count =count+1
        st_dev[j] =math.sqrt(st_dev[j]/float(used_smpl))
        j=j+1
    #***************************************************************************************************************************
    """close all the txt files """
   # open('wtpopulation.txt', 'w').close()
    open('wtresult.txt', 'w').close()
    open('wtresult1.txt', 'w').close()
    open('wtresultnew.txt', 'w').close()
   # open('wt_onlyresult.txt', 'w').close()
    open('wt_onlyresult_append.txt', 'w').close()
    open('out.txt', 'w').close()
    #***************************************************************************************************************************
    mu =mean_vector
    sigma =st_dev
    iteration =iteration+1
