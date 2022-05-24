import os
import time
import nsfw_m as nsfw_module
import csv

test_directory='/home/w5fast1'
model=nsfw_module.load_model()
image_counter=0
t0=time.time()
for subject in os.listdir(test_directory):
    if subject.startswith('w5'):
        filename = subject + '-nsfwresults.csv'
        with open(os.path.join(test_directory,filename),'w+') as log:
            csv_writer = csv.writer(log)
            for screenshot in os.listdir(os.path.join(test_directory,subject)):
                try: output = nsfw_module.predict(os.path.join(test_directory,subject, screenshot),model)
                except: pass
                else:
                    image_counter+=1
                    for key, value in output.items():
                        csv_writer.writerow([key,value])  
                    if(image_counter>5):  
                        break
        t1=time.time()
        total_elapsed=t1-t0
        with open('summary_time_elapse.txt', 'a+') as f:
            f.write(subject+" "+ str(image_counter) + " images "+str(total_elapsed)+" seconds\n")   
    else: continue