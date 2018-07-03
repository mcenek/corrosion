###########################################################################################
#This parsing python script takes a file created from execute.py and parses it so that
#it can be used for joachim's svm.
#
###########################################################################################
###########################################################################################
#Library imports
import sys
import re

if len(sys.argv) == 4 and (sys.argv[3] == '+' or sys.argv[3] == '-'):
    filenamein = sys.argv[1]
    filenameout = sys.argv[2]
    classification = sys.argv[3]

    if len(filenamein) == 0 or len(filenameout) == 0:
        print("bad files passed")
        print "expected: "
        print "file_directory_in file_directory_out"
        sys.exit()

    with open(filenamein, 'r') as fin, open(filenameout,'w') as fout:
        lines = fin.read().splitlines()
        for l in lines:
            tokens = re.findall("\d+\.\d+",l)
            fout.write(str(classification) + "1 ")
            for i,t in enumerate(tokens):
                fout.write(str(i + 1))
                fout.write(":")
                fout.write(str(t))
                fout.write(" ")
            fout.write("\n")
else:
    print "error with the number of arguments:"
    print "argv1 = file_name_in"
    print "argv2 = file_name_out"
    print "argv3 = classification (+/-)"
