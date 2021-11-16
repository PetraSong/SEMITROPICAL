import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='graphbuilder for NN output')
parser.add_argument('-infile',metavar='I',default="",help='path to input file: ',required=True)
parser.add_argument('-outfolder',metavar='OutFolder',default="graphbuilder_out/",help='path of output folder to put plots: ',required=False)
parser.add_argument('-prefix', metavar='PREFIX',default="",help='prefix to give images', required=False)
args = parser.parse_args()

f = open(args.infile, "r")
losslist = []
acclist = []
valacclist = []
vallosslist = []

counter = 0
reading = False
for i in f:
    if "Epoch" in i:
        f.readline()
        if f.readline() == '\n':
            reading = True
    if reading == True:
        if i == '\n':
            reading=False
            counter += 1
            plt.plot(losslist)
            plt.plot(acclist)
            plt.title('epoch '+str(counter)+' loss and accuracy vs. steps')
            plt.ylabel('accuracy and loss')
            plt.xlabel('n steps')
            plt.legend(['loss', 'accuracy'], loc='upper left')
            print("saving plot number: " + str(counter)+"...")
            plt.savefig(args.outfolder+args.prefix+"_plotnumber_"+str(counter)+".png", bbox_inches='tight')
            plt.clf()
            losslist = []
            acclist = []
        else:
            loss_index = i.find("loss:",0)
            if loss_index > 0:
                loss = i[loss_index+6:loss_index+12]
                losslist.append(float(loss))
            acc_index = i.find("acc:",0)
            if acc_index > 0:    
                acc=i[acc_index+5:acc_index+11]
                acclist.append(float(acc))
            val_loss_index = i.find("val_loss:")
            if val_loss_index > 0:
                val_loss=i[val_loss_index+10:val_loss_index+16]
                vallosslist.append(float(val_loss))
            val_acc_index=i.find("val_acc:")
            if val_acc_index > 0:
                val_acc=i[val_acc_index+9:val_acc_index+15]
                valacclist.append(float(val_acc))

plt.plot(vallosslist)
plt.plot(valacclist)
plt.title('validation loss and accuracy vs. epochs')
plt.ylabel('validation accuracy and loss')
plt.xlabel('n epochs')
plt.legend(['val_loss', 'val_accuracy'], loc='upper left')
print("saving validation plot...")
plt.savefig(args.outfolder+args.prefix+"_validation.png", bbox_inches='tight')
plt.clf()
