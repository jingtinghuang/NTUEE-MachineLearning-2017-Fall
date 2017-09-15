import sys


def  main(argv):
    words = open(argv[0], "r").read().split()  # read the words into a list.
    l = []
    for c in words:
        if c not in l:
            l.append(c)
    text_file = open("./Q1.txt", "w")
    c = 0
    for n in range(len(l)):
        if n != len(l)-1:
            text_file.write(str(l[n])+" "+str(c)+" "+str(words.count(l[n]))+"\n")
        else:
            text_file.write(str(l[n]) + " " + str(c) + " " + str(words.count(l[n])))
        c += 1
    text_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])